# Iteration-by-iteration trace of ONE CUTEst problem under the best F64 penalty config
# (tE10_k2_ts1): CoshKernel + linear κ_P=2 StaticContinuation, condensed KKT, DEFAULT linear
# solver + DEFAULT scaling + tol=1e-8 — i.e. reproduces run_penalty_diag.jl's exact madnlp call,
# but with the lab Recorder attached so every iterate's state is dumped.
#
# Usage: julia --project=bench/penalty bench/penalty_lab/trace_one.jl HS7
# Writes results/trace/<NAME>__tE10_k2.csv (full per-iteration trajectory).

include(joinpath(@__DIR__, "harness.jl"))      # Recorder, eqfeas, _COLS
using MadNLP, CUTEst, NLPModels

const NAME = length(ARGS) >= 1 ? ARGS[1] : "HS7"

# tE10_k2_ts1 schedule: linear (θ_P=1) κ_P=2, E-gate τ_E=10 (ε_P=10/μ_P), s-gate τ_s=1, muP0=2.
sched = MadNLP.StaticContinuation(kappa_P = 2.0, theta_P = 1.0,
                                  opt_tol_coef = 10.0, opt_tol_exp = -1.0,
                                  s_thresh = 1.0, muP_max = 1.0e7)
treat = KernelPenaltyEquality(CoshKernel(); schedule = sched, muP = 2.0)

nlp = CUTEstModel(NAME; decode = false)
rec = Recorder()
local r
t = @elapsed r = madnlp(nlp;
        kkt_system            = MadNLP.SparseCondensedKKTSystem,   # default solver + default scaling
        tol                   = 1e-8,
        acceptable_tol        = 1e-8,
        equality_treatment    = treat,
        intermediate_callback = rec,
        max_iter              = 3000,
        print_level           = MadNLP.ERROR)

ef = eqfeas(nlp, r.solution)
println("TRACE $NAME: status=$(r.status) iters=$(r.iter) nfact=$(r.counters.factorization_cnt) "*
        "time=$(round(t,digits=2))s eqfeas=$(ef) obj=$(NLPModels.obj(nlp, r.solution)) "*
        "muP_final=$(Float64(treat.muP[])) x*=$(round.(Float64.(r.solution),digits=6))")

outdir = joinpath(@__DIR__, "results", "trace"); mkpath(outdir)
csv = joinpath(outdir, "$(NAME)__tE10_k2.csv")
open(csv, "w") do io
    println(io, join(_COLS, ","))
    for row in rec.rows
        println(io, join((row[c] for c in _COLS), ","))
    end
end
println("wrote $csv  ($(length(rec.rows)) iterations)")
finalize(nlp)
