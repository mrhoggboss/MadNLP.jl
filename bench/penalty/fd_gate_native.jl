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
using MadNLP, MadNLPHSL, CUTEst, NLPModels, LinearAlgebra, Printf, Random

const LINEAR_SOLVER = Ma27Solver               # augmented-path solver for the KKT-agnostic demos
const KKT_SYSTEM    = MadNLP.SparseKKTSystem

# FD-gate KKT configs: (label, kkt_system, linear_solver). The condensed/CHOLMOD path is SPD,
# and CHOLMOD is a Cholesky solver — so factorizing it at all *proves* the condensed matrix is
# SPD (the inertia claim). The condensed path is the cuDSS-GPU path in a later stage.
const CONFIGS = [
    (label = "augmented/MA27",    kkt = MadNLP.SparseKKTSystem,          ls = Ma27Solver),
    (label = "condensed/CHOLMOD", kkt = MadNLP.SparseCondensedKKTSystem, ls = MadNLP.CHOLMODSolver),
]

# Build a solver with the penalty treatment and initialize it (no solve), for white-box
# inspection of the assembled quantities.
function build_solver(nlp; muP, kkt = KKT_SYSTEM, ls = LINEAR_SOLVER, kernel = QuadraticKernel())
    solver = MadNLP.MadNLPSolver(nlp;
        equality_treatment = KernelPenaltyEquality(kernel; muP = muP),
        linear_solver = ls,
        kkt_system    = kkt,
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

# gradient: directional central FD of the total objective over the eq slacks, along one
# fixed pseudo-random direction d, vs dot(slack(f)[es], d) (slack(f) = φ'). Also checks all
# non-eq slack entries of f are zero. The directional form needs only O(1) eval_f calls (not
# O(m_eq)) and normalizes by the aggregate derivative, so it stays accurate at THRESH=1e-8 on
# large / many-equality problems (no per-component tiny-denominator blowup, less cancellation).
# ε is large because the QuadraticKernel has zero central-FD truncation error, so a bigger step
# only reduces roundoff against the (possibly large) total objective.
function check_gradient(solver; ε = 1e-3, seed = 1)
    cb = MadNLP.get_cb(solver); eh = cb.equality_handler
    x = MadNLP.get_x(solver); f = MadNLP.get_f(solver)
    es = eh.ind_eqslack
    MadNLP.eval_grad_f_wrapper!(solver, f, x)
    sf = MadNLP.slack(f); s = MadNLP.slack(x)
    isempty(es) && return (0.0, all(iszero, sf))
    ok_other = all(iszero, @view sf[setdiff(1:length(sf), es)])
    d  = randn(Random.MersenneTwister(seed), length(es))
    s0 = copy(@view s[es])
    @views s[es] .= s0 .+ ε .* d; fp = MadNLP.eval_f_wrapper(solver, x)
    @views s[es] .= s0 .- ε .* d; fm = MadNLP.eval_f_wrapper(solver, x)
    @views s[es] .= s0
    fd  = (fp - fm) / (2ε)
    ana = dot(@view(sf[es]), d)
    return (abs(fd - ana) / (abs(ana) + 1e-30), ok_other)
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

# Set the equality slacks to a non-degenerate O(1) point. The initial iterate may be exactly
# feasible (penalty ≈ 0, e.g. DIXCHLNG/ELEC) or wildly infeasible (penalty ~1e16, e.g.
# CATENARY); either makes the *relative* objective/gradient checks roundoff-dominated.
# Evaluating the penalty machinery at a clean O(1) slack point makes the checks meaningful.
# Call AFTER check_structure, which needs the pristine initial coupling residual.
function set_test_slacks!(solver)
    eh = MadNLP.get_cb(solver).equality_handler
    eh isa KernelPenaltyEquality || return
    es = eh.ind_eqslack; isempty(es) && return
    s = MadNLP.slack(MadNLP.get_x(solver))
    @views s[es] .= 0.2 .+ 0.6 .* (1:length(es)) ./ length(es)   # spread over (0.2, 0.8]
    return
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
# Spanning constraint types {eq, eq+bounds, eq+ineq+bounds, none} × objective convexity
# {linear, convex quadratic, general convex, nonconvex} × size (2 → 10000 vars). Only mE>0
# problems exercise the penalty; the no-equality ones verify it cleanly no-ops.
const FD_PROBS = [
    # ── nonconvex objective ──
    "HS6", "HS7", "HS8", "HS9", "HS40", "HS56", "HS78",     # eq only (nonlinear eq), small
    "DIXCHLNG", "ORTHREGB", "MSS1", "ELEC", "LUKVLE3",      # eq only, medium → large (10000)
    "HS60", "HS63", "HS107", "HS119", "CATENARY", "DTOC2",  # eq + bounds, small → large
    "HS14", "HS71", "HS114",                                # eq + inequality + bounds (mixed)
    # ── convex quadratic objective ──
    "HS28", "HS52", "GENHS28",                              # eq only (linear eq)
    "HS53", "DUAL1",                                        # eq + bounds (DUAL1: 85 vars)
    # ── general convex objective (log / entropy) ──
    "HS111", "HS112",                                       # eq + bounds
    # ── linear objective ──
    "EXTRASIM",                                             # eq + bound
    # ── no equalities → penalty must cleanly no-op (mE = 0) ──
    "HS21", "HS118", "AVGASB",                              # convex-quad / linear obj, ineq + bounds
]
# CPU accuracy target (matches the solver tol we run at). Stage 4 (GPU/cuDSS) should validate
# at 1e-6 instead.
const THRESH    = 1e-8

println("=== FD GATE: native KernelPenaltyEquality (QuadraticKernel), per KKT config ===\n")
function run_fd_gate(cfg)
    @printf("[%s]\n", cfg.label)
    @printf("%-7s %5s %5s | %-7s %-9s %-9s %-9s %-6s\n",
            "prob", "n", "mE", "struct", "obj", "grad", "diag", "g0?")
    fd_ok = true
    for p in FD_PROBS
        nlp = CUTEstModel(p; decode = true)
        try
            solver = build_solver(nlp; muP = 50.0, kkt = cfg.kkt, ls = cfg.ls)
            eh = MadNLP.get_cb(solver).equality_handler
            st  = check_structure(solver)      # pristine initial point (coupling residual, bounds)
            set_test_slacks!(solver)           # then move eq-slacks to an O(1) point
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
function run_all_fd_gates()
    ok = true
    for cfg in CONFIGS
        ok &= run_fd_gate(cfg); println()
    end
    return ok
end
fd_ok = run_all_fd_gates()
println(fd_ok ? "FD_GATE_PASS" : "FD_GATE_FAIL")

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
                     schedule = StaticContinuation(kappa_P = 10.0, theta_P = 1.5, s_thresh = 1e-2, muP_max = 1e6), muP = 1.0),
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

# ---- Stage 3: condensed (SPD) KKT path -----------------------------------------------
# The penalty D_s rides pr_diag → build_kkt!'s diag_buffer with NO new injection code, so the
# treatment also works on SparseCondensedKKTSystem. CHOLMOD (Cholesky) factorizing it proves
# the condensed matrix is SPD. This is the cuDSS-GPU path in the next stage.
println("\n=== Stage 3: condensed KKT path (continuation), per config ===")
let
    nlp = CUTEstModel("HS14"; decode = true)
    try
        for cfg in CONFIGS
            r = madnlp(nlp; equality_treatment = KernelPenaltyEquality(QuadraticKernel();
                       schedule = StaticContinuation(kappa_P = 10.0, theta_P = 1.5, s_thresh = 1e-2, muP_max = 1e6), muP = 1.0),
                       kkt_system = cfg.kkt, linear_solver = cfg.ls,
                       nlp_scaling = false, print_level = MadNLP.ERROR)
            ok = r.status in (MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL)
            @printf("%-18s : %-26s iters=%d eqfeas=%.2e obj=% .6e %s\n",
                    cfg.label, r.status, r.iter, eq_feas(nlp, r.solution[1:get_nvar(nlp)]),
                    obj(nlp, r.solution[1:get_nvar(nlp)]),
                    ok ? "" : "<-- EXPECTED CONVERGENCE")
        end
    finally
        finalize(nlp)
    end
end

println("\nGATE_SCRIPT_DONE")
