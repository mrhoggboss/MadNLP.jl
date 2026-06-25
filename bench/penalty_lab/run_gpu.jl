# μ_P continuation lab — GPU DRIVER (condensed + cuDSS).
# (Responsibility: experiment SPECIFICATION + execution on the GPU. Reuses the bench/penalty_gpu
#  env. Same harness/schedules/report as the CPU driver — only the problem source, kkt_system,
#  and linear_solver differ. CUTEst is CPU-only, and MadNLPTests.DenseDummyQP scalar-indexes its
#  Jacobian on the GPU, so GPU configs use ExaModels, whose cons/jac/hess are GPU kernels.)
#
# Run:  julia bench/penalty_lab/run_gpu.jl

import Pkg; Pkg.activate(joinpath(@__DIR__, "..", "penalty_gpu"); io = devnull)
using CUDA, CUDSS, MadNLP, MadNLPGPU, ExaModels
import NLPModels

include(joinpath(@__DIR__, "schedules.jl"))
include(joinpath(@__DIR__, "harness.jl"))
include(joinpath(@__DIR__, "report.jl"))

@assert CUDA.functional() "CUDA is not functional"
@info "GPU device: $(CUDA.name(CUDA.device()))"

const RESULTS = joinpath(@__DIR__, "results", "gpu")

# A device-resident mixed convex problem (ExaModels → GPU kernels for cons/jac/hess):
#   min Σ (xᵢ-2)²   s.t.  xᵢ+xᵢ₊₁ = 3 (equalities),  xᵢ² ≤ 4 (inequalities),  0 ≤ xᵢ ≤ 5.
function mixed_exa(N; backend)
    c = ExaCore(Float64; backend = backend)
    x = variable(c, N; lvar = 0.0, uvar = 5.0, start = 1.0)
    objective(c, (x[i] - 2.0)^2 for i = 1:N)
    constraint(c, x[i] + x[i+1] - 3.0 for i = 1:N-1)                  # equalities (lcon=ucon=0)
    constraint(c, x[i]^2 - 4.0 for i = 1:N; lcon = -Inf, ucon = 0.0) # inequalities
    return ExaModel(c)
end
exa(N) = (make_nlp = () -> mixed_exa(N; backend = CUDA.CUDABackend()), cleanup = _ -> nothing)

const CON_GPU = (kkt = MadNLP.SparseCondensedKKTSystem, ls = MadNLPGPU.CUDSSSolver)

cfg(name, prob, sys; kernel = QuadraticKernel(), schedule = FixedPenalty(), muP = 1.0,
    tol = 1e-6, max_iter = 3000) = (         # GPU accuracy target is 1e-6
    name      = name,
    make_nlp  = prob.make_nlp,
    cleanup   = prob.cleanup,
    kkt       = sys.kkt,
    ls        = sys.ls,
    treatment = KernelPenaltyEquality(kernel; schedule = schedule, muP = muP),
    tol       = tol,
    max_iter  = max_iter,
)

# ======================= GPU EXPERIMENT CONFIGS (edit me) =======================
P = exa(50)
CONFIGS = [
    cfg("exa_con_fixed_1e3", P, CON_GPU; muP = 1e3),
    cfg("exa_con_static_r1", P, CON_GPU; schedule = StaticContinuation(rho = 1.0, muP_max = 1e6)),
]
# ================================================================================

results = RunResult[]
for c in CONFIGS
    r = run_config(c)
    push!(results, r)
    save_trajectory(r, RESULTS)
    println(stderr, "ran $(c.name): $(r.status), $(r.iters) iters, eqfeas=$(r.eqfeas)")
end
save_summary(results, RESULTS)
println("\n=== SUMMARY (GPU / condensed / cuDSS) ===")
summary_table(results)
println("\ntrajectories + summary.csv written to: $RESULTS")
println("LAB_GPU_RUN_DONE")
