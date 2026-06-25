# μ_P continuation lab — CPU DRIVER (augmented/MA27 + condensed/CHOLMOD).
# (Responsibility: experiment SPECIFICATION + execution. Reuses the bench/penalty env. Edit
#  CONFIGS to match what you describe; schedules live in schedules.jl, plumbing in harness.jl,
#  reporting in report.jl.)
#
# Run:  julia bench/penalty_lab/run_cpu.jl

import Pkg; Pkg.activate(joinpath(@__DIR__, "..", "penalty"); io = devnull)
using MadNLP, MadNLPHSL, CUTEst, NLPModels

include(joinpath(@__DIR__, "schedules.jl"))
include(joinpath(@__DIR__, "harness.jl"))
include(joinpath(@__DIR__, "report.jl"))

const RESULTS = joinpath(@__DIR__, "results", "cpu")

# problem source: a CUTEst model (must be finalized after each run)
cutest(name) = (make_nlp = () -> CUTEstModel(name; decode = true), cleanup = finalize)

# CPU KKT systems
const AUG = (kkt = MadNLP.SparseKKTSystem,          ls = Ma27Solver)
const CON = (kkt = MadNLP.SparseCondensedKKTSystem, ls = MadNLP.CHOLMODSolver)

# config builder: name × problem × system × treatment(kernel, schedule, μ_P0)
cfg(name, prob, sys; kernel = QuadraticKernel(), schedule = FixedPenalty(), muP = 1.0,
    tol = 1e-8, max_iter = 3000) = (
    name      = name,
    make_nlp  = prob.make_nlp,
    cleanup   = prob.cleanup,
    kkt       = sys.kkt,
    ls        = sys.ls,
    treatment = KernelPenaltyEquality(kernel; schedule = schedule, muP = muP),
    tol       = tol,
    max_iter  = max_iter,
)

# ======================= EXPERIMENT CONFIGS (edit me) =======================
HS14 = cutest("HS14")
CONFIGS = [
    cfg("hs14_aug_fixed_1e3",  HS14, AUG; muP = 1e3),
    cfg("hs14_aug_static_r1",  HS14, AUG; schedule = StaticContinuation(kappa_P = 10.0, theta_P = 1.5, muP_max = 1e6)),
    cfg("hs14_con_fixed_1e3",  HS14, CON; muP = 1e3),
    cfg("hs14_con_static_r1",  HS14, CON; schedule = StaticContinuation(kappa_P = 10.0, theta_P = 1.5, muP_max = 1e6)),
]
# ============================================================================

results = RunResult[]
for c in CONFIGS
    r = run_config(c)
    push!(results, r)
    save_trajectory(r, RESULTS)
    println(stderr, "ran $(c.name): $(r.status), $(r.iters) iters, eqfeas=$(r.eqfeas)")
end
save_summary(results, RESULTS)
println("\n=== SUMMARY ===")
summary_table(results)
println("\ntrajectories + summary.csv written to: $RESULTS")
println("LAB_RUN_DONE")
