# Stage-1 validation gate for the native KernelPenaltyEquality treatment.
#
# Verifies the penalty injection MadNLP performs internally (objective, slack gradient,
# pr_diag diagonal, bound-index exclusion) against finite differences and analytic values,
# on small mixed (equality + inequality/bounds) CUTEst problems, plus an end-to-end
# convergence check. Parameterized over `linear_solver`/`kkt_system` so the future
# condensed/cuDSS-GPU path is a config change (MA27 + SparseKKTSystem for now).
#
# Run:  julia --project=bench/penalty bench/penalty/fd_gate_native.jl

import Pkg; Pkg.activate(@__DIR__; io = devnull)
using MadNLP, MadNLPHSL, CUTEst, NLPModels, LinearAlgebra, Printf

const LINEAR_SOLVER = Ma27Solver               # swap to a cuDSS solver on the GPU path later
const KKT_SYSTEM    = MadNLP.SparseKKTSystem    # swap to SparseCondensedKKTSystem later

# Build a solver with the penalty treatment and initialize it (no solve), for white-box
# inspection of the assembled quantities.
function build_solver(nlp; muP, kernel = QuadraticKernel())
    solver = MadNLP.MadNLPSolver(nlp;
        equality_treatment = KernelPenaltyEquality(kernel; muP = muP),
        linear_solver = LINEAR_SOLVER,
        kkt_system    = KKT_SYSTEM,
        nlp_scaling   = false,          # remove the scaling-units confound for the FD gate
        print_level   = MadNLP.ERROR,
    )
    MadNLP.initialize!(solver)
    return solver
end

# ---- white-box checks (return relative/abs errors; smaller is better) ----------------

function check_structure(solver)
    cb = MadNLP.get_cb(solver); eh = cb.equality_handler
    x  = MadNLP.get_x(solver)
    n  = length(MadNLP.variable(x))
    es = eh.ind_eqslack
    slackpos = n .+ es                                  # eq-slack positions in the full primal
    ok_lb = !any(in(cb.ind_lb), slackpos)               # excluded from bound-barrier sets
    ok_ub = !any(in(cb.ind_ub), slackpos)
    xl = MadNLP.full(MadNLP.get_xl(solver)); xu = MadNLP.full(MadNLP.get_xu(solver))
    ok_bounds = all(xl[slackpos] .== -Inf) && all(xu[slackpos] .== Inf)
    ok_rhs = solver.rhs[es] == eh.b                     # equality target preserved (raw, scaling off)
    # initial coupling residual c(x0) - s - rhs on equality rows
    MadNLP.eval_cons_wrapper!(solver, MadNLP.get_c(solver), x)
    ok_couple = isempty(es) ? true : norm(MadNLP.get_c(solver)[es], Inf) < 1e-10
    return ok_lb && ok_ub && ok_bounds && ok_rhs && ok_couple
end

# objective: eval_f_wrapper - base == Σ φ(μ, s_eq)
function check_objective(solver)
    cb = MadNLP.get_cb(solver); eh = cb.equality_handler
    x = MadNLP.get_x(solver); μ = eh.muP[]; es = eh.ind_eqslack
    isempty(es) && return 0.0
    s = MadNLP.slack(x)
    base  = MadNLP._eval_f_wrapper(cb, MadNLP.variable(x))   # minimize ⇒ sense = +1
    total = MadNLP.eval_f_wrapper(solver, x)
    pen_expected = sum(MadNLP.kernel_val(eh.kernel, μ, s[i]) for i in es)
    return abs((total - base) - pen_expected) / (abs(pen_expected) + 1e-30)
end

# gradient: central FD of total objective wrt each eq slack == slack(f)[es] (= φ'); and all
# non-eq slack entries of f are zero.
function check_gradient(solver; ε = 1e-6)
    cb = MadNLP.get_cb(solver); eh = cb.equality_handler
    x = MadNLP.get_x(solver); f = MadNLP.get_f(solver)
    es = eh.ind_eqslack
    MadNLP.eval_grad_f_wrapper!(solver, f, x)
    sf = MadNLP.slack(f); s = MadNLP.slack(x)
    isempty(es) && return (0.0, all(sf .== 0))
    others = setdiff(1:length(sf), es)
    ok_other = all(sf[others] .== 0)
    rel = 0.0
    for i in es
        s0 = s[i]
        s[i] = s0 + ε; fp = MadNLP.eval_f_wrapper(solver, x)
        s[i] = s0 - ε; fm = MadNLP.eval_f_wrapper(solver, x)
        s[i] = s0
        fd = (fp - fm) / (2ε)
        rel = max(rel, abs(fd - sf[i]) / (abs(sf[i]) + 1e-30))
    end
    return (rel, ok_other)
end

# assembled diagonal: pr_diag[n+es] == reg + D_s(μ, s_eq)
function check_diagonal(solver)
    cb = MadNLP.get_cb(solver); eh = cb.equality_handler; kkt = MadNLP.get_kkt(solver)
    x = MadNLP.get_x(solver); μ = eh.muP[]; es = eh.ind_eqslack
    isempty(es) && return 0.0
    n = length(MadNLP.variable(x))
    MadNLP.set_aug_diagonal!(kkt, solver)
    reg = MadNLP.get_opt(solver).default_primal_regularization
    s = MadNLP.slack(x)
    expected = [reg + MadNLP.kernel_hess(eh.kernel, μ, s[i]) for i in es]
    return norm(kkt.pr_diag[n .+ es] .- expected, Inf)
end

# ---- end-to-end equality feasibility -------------------------------------------------

function eq_feas(nlp, x)
    m = get_ncon(nlp); m == 0 && return 0.0
    c = zeros(m); cons!(nlp, x, c)
    lc = get_lcon(nlp); uc = get_ucon(nlp)
    f = 0.0
    for i in 1:m
        lc[i] == uc[i] && (f = max(f, abs(c[i] - lc[i])))
    end
    return f
end

# ============================ run ============================
const FD_PROBS  = ["HS6", "HS7", "HS8", "HS9", "HS28", "HS48", "HS52", "HS14"]
const THRESH    = 1e-6

println("=== Stage-1 FD GATE: native KernelPenaltyEquality (QuadraticKernel) ===")
println("linear_solver=$(LINEAR_SOLVER)  kkt_system=$(KKT_SYSTEM)\n")
function run_fd_gate()
    @printf("%-7s %5s %5s | %-7s %-9s %-9s %-9s %-6s\n",
            "prob", "n", "mE", "struct", "obj", "grad", "diag", "g0?")
    fd_ok = true
    for p in FD_PROBS
        nlp = CUTEstModel(p; decode = true)
        try
            solver = build_solver(nlp; muP = 50.0)
            eh = MadNLP.get_cb(solver).equality_handler
            st  = check_structure(solver)
            ov  = check_objective(solver)
            (gr, g0) = check_gradient(solver)
            dg  = check_diagonal(solver)
            pass = st && (ov < THRESH) && (gr < THRESH) && (dg < 1e-10) && g0
            fd_ok &= pass
            @printf("%-7s %5d %5d | %-7s %.2e %.2e %.2e %-6s %s\n",
                    p, get_nvar(nlp), length(eh.ind_eqslack),
                    st ? "ok" : "FAIL", ov, gr, dg, g0 ? "ok" : "BAD", pass ? "" : " <-- FAIL")
        catch e
            fd_ok = false
            @printf("%-7s  ERROR: %s\n", p, sprint(showerror, e)[1:min(140, end)])
        finally
            finalize(nlp)
        end
    end
    return fd_ok
end
fd_ok = run_fd_gate()
println(fd_ok ? "\nFD_GATE_PASS" : "\nFD_GATE_FAIL")

# end-to-end: HS14 equality feasibility should fall ~1/μ (quadratic penalty) and the
# objective should approach the exact EnforceEquality solution as μ grows.
println("\n=== End-to-end: HS14, fixed μ sweep vs EnforceEquality ===")
let
    nlp = CUTEstModel("HS14"; decode = true)
    try
        rref = madnlp(nlp; equality_treatment = MadNLP.EnforceEquality, linear_solver = LINEAR_SOLVER,
                      nlp_scaling = false, print_level = MadNLP.ERROR, tol = 1e-8)
        @printf("EnforceEquality : status=%s  obj=% .8e  eqfeas=%.2e\n",
                rref.status, rref.objective, eq_feas(nlp, rref.solution[1:get_nvar(nlp)]))
        for μ in (1e1, 1e3, 1e5, 1e7)
            r = madnlp(nlp; equality_treatment = KernelPenaltyEquality(QuadraticKernel(); muP = μ),
                       linear_solver = LINEAR_SOLVER, kkt_system = KKT_SYSTEM,
                       nlp_scaling = false, print_level = MadNLP.ERROR, tol = 1e-8)
            x = r.solution[1:get_nvar(nlp)]
            @printf("Penalty μ=%.0e : status=%s  obj=% .8e  eqfeas=%.2e  Δobj=%.2e\n",
                    μ, r.status, obj(nlp, x), eq_feas(nlp, x), abs(obj(nlp, x) - rref.objective))
        end
    finally
        finalize(nlp)
    end
end
# overflow policy: plain (non-saturating) cosh must TERMINATE the solve with a clear
# message when μ·s is large enough to overflow, not silently saturate.
println("\n=== Overflow policy: CoshKernel must fail loudly, not saturate ===")
let
    nlp = CUTEstModel("HS14"; decode = true)
    try
        rok = madnlp(nlp; equality_treatment = KernelPenaltyEquality(CoshKernel(); muP = 1.0),
                     linear_solver = LINEAR_SOLVER, nlp_scaling = false,
                     print_level = MadNLP.ERROR, tol = 1e-8)
        @printf("Cosh μ=1     : status=%s (moderate μ solves without overflow)\n", rok.status)
        println(">>> expect a penalty-overflow @error + INVALID_NUMBER status below:")
        rbig = madnlp(nlp; equality_treatment = KernelPenaltyEquality(CoshKernel(); muP = 1e4),
                      linear_solver = LINEAR_SOLVER, nlp_scaling = false,
                      print_level = MadNLP.ERROR, tol = 1e-8)
        invalid = (MadNLP.INVALID_NUMBER_OBJECTIVE, MadNLP.INVALID_NUMBER_GRADIENT,
                   MadNLP.INVALID_NUMBER_HESSIAN_LAGRANGIAN, MadNLP.INVALID_NUMBER_DETECTED)
        @printf("Cosh μ=1e4   : status=%s  %s\n", rbig.status,
                rbig.status in invalid ? "(terminated on overflow — OK)" : "<-- EXPECTED INVALID_NUMBER")
    finally
        finalize(nlp)
    end
end

# ---- Stage 2: μ_P continuation schedules (decoupled from μ_B) -----------------------
# A user-defined schedule whose μ_P depends ONLY on the iteration count, never on μ_B —
# proof that the schedule infrastructure is decoupled and that the filter correctly resets
# on μ_P moves (otherwise this would stall). Defined entirely in user code.
struct StepEveryK <: MadNLP.AbstractPenaltySchedule
    factor::Float64
    every::Int
    muP_max::Float64
end
function MadNLP.update_penalty!(s::StepEveryK, eh, solver)
    T = typeof(eh.muP[]); k = MadNLP.get_cnt(solver).k
    (k > 0 && k % s.every == 0) && (eh.muP[] = min(T(s.muP_max), eh.muP[] * T(s.factor)))
    return
end

println("\n=== Stage 2: μ_P continuation (StaticContinuation + a decoupled user schedule) ===")
let
    nlp = CUTEstModel("HS14"; decode = true)
    try
        # fixed μ at the same ceiling stalls (Stage-1 conditioning wall); continuation converges.
        rfix = madnlp(nlp; equality_treatment = KernelPenaltyEquality(QuadraticKernel(); muP = 1e6),
                      linear_solver = LINEAR_SOLVER, nlp_scaling = false, print_level = MadNLP.ERROR)
        @printf("Quad fixed μ=1e6           : %-26s iters=%d\n", rfix.status, rfix.iter)
        rsc = madnlp(nlp; equality_treatment = KernelPenaltyEquality(QuadraticKernel();
                     schedule = StaticContinuation(rho = 1.0, muP_max = 1e6), muP = 1.0),
                     linear_solver = LINEAR_SOLVER, nlp_scaling = false, print_level = MadNLP.ERROR)
        @printf("Quad StaticContinuation    : %-26s iters=%d  eqfeas=%.2e\n",
                rsc.status, rsc.iter, eq_feas(nlp, rsc.solution[1:get_nvar(nlp)]))
        rdk = madnlp(nlp; equality_treatment = KernelPenaltyEquality(QuadraticKernel();
                     schedule = StepEveryK(3.0, 5, 1e6), muP = 1.0),
                     linear_solver = LINEAR_SOLVER, nlp_scaling = false, print_level = MadNLP.ERROR)
        ok = rdk.status in (MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL)
        @printf("Decoupled StepEveryK (∌μ_B): %-26s iters=%d  eqfeas=%.2e  %s\n",
                rdk.status, rdk.iter, eq_feas(nlp, rdk.solution[1:get_nvar(nlp)]),
                ok ? "(decoupled schedule converged — OK)" : "<-- EXPECTED CONVERGENCE")
    finally
        finalize(nlp)
    end
end

println("\nGATE_SCRIPT_DONE")
