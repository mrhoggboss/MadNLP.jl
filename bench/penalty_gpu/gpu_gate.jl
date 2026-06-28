# Stage-4 GPU validation for the native KernelPenaltyEquality treatment.
#
# The penalty rides pr_diag → build_kkt!'s diag_buffer (condensed path) and the kernels +
# injections are GPU-broadcast-safe, so the treatment should run on the GPU condensed/cuDSS
# path with no new code. This checks exactly that: no scalar indexing, results match the CPU
# solve, at tol = 1e-6 (the GPU accuracy target).
#
# Run:  julia --project=bench/penalty_gpu bench/penalty_gpu/gpu_gate.jl

import Pkg; Pkg.activate(@__DIR__; io = devnull)
using CUDA, CUDSS, MadNLP, MadNLPGPU, MadNLPTests, NLPModels, LinearAlgebra, Printf

@assert CUDA.functional() "CUDA is not functional"
@printf("CUDA device: %s\n\n", CUDA.name(CUDA.device()))

const CO = MadNLP.SparseCondensedKKTSystem

# eq feasibility ‖(c-b) on equality rows‖∞, computed on-device (no scalar indexing)
function eqfeas(nlp, x)
    m = get_ncon(nlp); m == 0 && return 0.0
    c = similar(x, m); cons!(nlp, x, c)
    lc = get_lcon(nlp); uc = get_ucon(nlp)
    return Float64(maximum(abs.(c .- lc) .* (lc .== uc)))
end

# A mixed convex-QP: bounds on all vars, `eqc` rows are equalities (Ax=0), the rest are
# inequalities. `arr` selects host (Array) or device (CuArray). DenseDummyQP seeds its data,
# so the host and device instances are identical.
build(n, m, eqc, arr) =
    MadNLPTests.DenseDummyQP(arr(zeros(Float64, n)); m = m, equality_cons = arr(collect(eqc)))

solve_pen(nlp, ls; kernel = QuadraticKernel(), muP = 1.0, sched = StaticContinuation(rho=1.0, muP_max=1e6)) =
    madnlp(nlp;
        callback           = MadNLP.SparseCallback,
        kkt_system         = CO,
        linear_solver      = ls,
        equality_treatment = KernelPenaltyEquality(kernel; schedule = sched, muP = muP),
        nlp_scaling        = false,
        tol                = 1e-6,
        print_level        = MadNLP.ERROR,
    )

const N, M, EQC = 50, 12, (1, 4, 7, 10)

println("=== Baseline: GPU condensed/cuDSS works (default treatment) ===")
let
    nlp = build(N, M, EQC, CuArray)
    r = madnlp(nlp; callback=MadNLP.SparseCallback, kkt_system=CO,
               linear_solver=MadNLPGPU.CUDSSSolver, print_level=MadNLP.ERROR, tol=1e-6)
    @printf("default/RelaxEquality : %-26s obj=% .8e\n", r.status, r.objective)
end

println("\n=== Penalty on GPU (cuDSS) vs CPU (CHOLMOD), QuadraticKernel + continuation ===")
let
    rcpu = solve_pen(build(N, M, EQC, Array),  MadNLP.CHOLMODSolver)
    rgpu = solve_pen(build(N, M, EQC, CuArray), MadNLPGPU.CUDSSSolver)
    xg = Array(rgpu.solution); xc = rcpu.solution
    nlp_h = build(N, M, EQC, Array)
    @printf("CPU/CHOLMOD : %-26s obj=% .8e eqfeas=%.2e\n", rcpu.status, rcpu.objective, eqfeas(nlp_h, xc))
    @printf("GPU/cuDSS   : %-26s obj=% .8e eqfeas=%.2e\n", rgpu.status, rgpu.objective, eqfeas(build(N,M,EQC,CuArray), rgpu.solution))
    @printf("|obj_gpu - obj_cpu| = %.2e ;  ‖x_gpu - x_cpu‖∞ = %.2e\n",
            abs(rgpu.objective - rcpu.objective), norm(xg .- xc, Inf))
    ok = abs(rgpu.objective - rcpu.objective) < 1e-6 && norm(xg .- xc, Inf) < 1e-5
    println(ok ? "GPU_MATCHES_CPU: OK" : "GPU_MATCHES_CPU: <-- MISMATCH")
end

println("\n=== White-box: D_s injection lands in pr_diag on the GPU (copied to host) ===")
let
    nlp = build(N, M, EQC, CuArray)
    solver = MadNLP.MadNLPSolver(nlp; callback=MadNLP.SparseCallback, kkt_system=CO,
             linear_solver=MadNLPGPU.CUDSSSolver,
             equality_treatment=KernelPenaltyEquality(QuadraticKernel(); muP=50.0),
             nlp_scaling=false, print_level=MadNLP.ERROR)
    MadNLP.initialize!(solver)
    MadNLP.set_aug_diagonal!(MadNLP.get_kkt(solver), solver)
    eh = MadNLP.get_cb(solver).equality_handler
    n  = length(MadNLP.variable(MadNLP.get_x(solver)))
    es = Array(eh.ind_eqslack)
    pr = Array(MadNLP.get_kkt(solver).pr_diag)
    reg = MadNLP.get_opt(solver).default_primal_regularization
    ok = all(abs.(pr[n .+ es] .- (reg + 50.0)) .< 1e-10)
    @printf("pr_diag[n+es] == reg+μ (=%.1f): %s  (e.g. %s)\n", reg+50.0, ok, pr[n .+ es[1:min(3,end)]])
end

println("\n=== CoshKernel on GPU (broadcast-safe), moderate μ continuation ===")
let
    r = solve_pen(build(N, M, EQC, CuArray), MadNLPGPU.CUDSSSolver;
                  kernel = CoshKernel(), muP = 1.0, sched = StaticContinuation(rho=0.5, muP_max=1e2))
    @printf("Cosh GPU    : %-26s obj=% .8e eqfeas=%.2e\n",
            r.status, r.objective, eqfeas(build(N,M,EQC,CuArray), r.solution))
end

println("\n=== No scalar indexing in the penalty path (allowscalar(false)) ===")
let
    CUDA.allowscalar(false)
    try
        r = solve_pen(build(N, M, EQC, CuArray), MadNLPGPU.CUDSSSolver)
        @printf("allowscalar(false) solve : %-26s (no scalar indexing — OK)\n", r.status)
    catch e
        @printf("allowscalar(false) ERROR: %s\n", sprint(showerror, e)[1:min(200,end)])
    finally
        CUDA.allowscalar(true)
    end
end

println("\nGPU_GATE_DONE")
