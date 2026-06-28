# Trace HS7 (or any problem) under two μ_P-schedule variants, to test the two hypotheses about
# why the linear/E-gated config (tE10_k2) stalls on the dual residual:
#
#   (1) superlin : SAME E-gate + s-gate as tE10_k2, but SUPERLINEAR growth θ_P=2
#                  (μ_next = max(2·μ, μ²)).  Tests "is it the growth RATE?"
#   (2) bcoupled : LINEAR rate (κ_P=2), NO E-gate, NO s-gate — μ_P bumps in lockstep with the
#                  barrier: whenever μ_B is reduced (IPOPT condition E ≤ 10·μ_B), bump μ_P too.
#                  Tests "is it the E-GATE coupling?"  (μ_P bump helps inf_du fall ∝1/μ_P, which
#                  may keep E ≤ 10·μ_B satisfiable so μ_B can keep dropping.)
#
# Everything else identical to the tE10_k2 trace (CoshKernel, condensed, default solver+scaling,
# tol=1e-8, muP0=2). Usage: julia --project=bench/penalty bench/penalty_lab/trace_variants.jl [NAME]

include(joinpath(@__DIR__, "harness.jl"))      # Recorder, eqfeas, _COLS
using MadNLP, CUTEst, NLPModels, LinearAlgebra

const NAME = length(ARGS) >= 1 ? ARGS[1] : "HS7"

# ---- (1) E-gated schedule, parameterized growth (replicates run_penalty_diag.jl's rule) ----
mutable struct GateStatic <: MadNLP.AbstractPenaltySchedule
    kappa_P::Float64; theta_P::Float64; opt_tol_coef::Float64; opt_tol_exp::Float64
    s_thresh::Float64; muP_max::Float64
end
function MadNLP.update_penalty!(s::GateStatic, eh::KernelPenaltyEquality,
                                solver::MadNLP.AbstractMadNLPSolver{T}) where T
    μ = eh.muP[]
    μ >= s.muP_max && return
    μ_next = min(T(s.muP_max), max(T(s.kappa_P)*μ, μ^T(s.theta_P)))
    E  = MadNLP.get_inf_barrier(solver)
    sE = isempty(eh.ind_eqslack) ? zero(T) :
         norm(view(MadNLP.slack(MadNLP.get_x(solver)), eh.ind_eqslack), Inf)
    Eok = E  <= T(s.opt_tol_coef) * μ^T(s.opt_tol_exp)
    sok = sE <= T(s.s_thresh) / μ_next
    (Eok && sok) && (eh.muP[] = μ_next)
    return
end

# ---- (2) barrier-coupled schedule: bump μ_P (linear) iff μ_B was just reduced; no E-gate ----
mutable struct BarrierCoupled <: MadNLP.AbstractPenaltySchedule
    kappa_P::Float64; theta_P::Float64; muP_max::Float64
    last_muB::Base.RefValue{Float64}
end
BarrierCoupled(; kappa_P=2.0, theta_P=1.0, muP_max=1.0e7) =
    BarrierCoupled(kappa_P, theta_P, muP_max, Ref(NaN))
function MadNLP.update_penalty!(s::BarrierCoupled, eh::KernelPenaltyEquality,
                                solver::MadNLP.AbstractMadNLPSolver{T}) where T
    μB = Float64(MadNLP.get_mu(solver))                 # μ_B AFTER this iter's barrier update
    if !isnan(s.last_muB[]) && μB < s.last_muB[] && eh.muP[] < s.muP_max   # μ_B dropped ⇒ advance μ_P
        eh.muP[] = min(T(s.muP_max), max(T(s.kappa_P)*eh.muP[], eh.muP[]^T(s.theta_P)))
    end
    s.last_muB[] = μB
    return
end

function trace(tag, sched)
    nlp = CUTEstModel(NAME; decode=false)
    rec = Recorder()
    treat = KernelPenaltyEquality(CoshKernel(); schedule=sched, muP=2.0)
    local r
    t = @elapsed r = madnlp(nlp; kkt_system=MadNLP.SparseCondensedKKTSystem, tol=1e-8,
                            acceptable_tol=1e-8, equality_treatment=treat,
                            intermediate_callback=rec, max_iter=3000, print_level=MadNLP.ERROR)
    ef = eqfeas(nlp, r.solution)
    # μ_P live value: read from the materialized handler via the last recorded row
    muPf = isempty(rec.rows) ? NaN : rec.rows[end].muP
    println("[$tag] $NAME: status=$(r.status) iters=$(r.iter) nfact=$(r.counters.factorization_cnt) "*
            "time=$(round(t,digits=2))s eqfeas=$(ef) obj=$(round(NLPModels.obj(nlp,r.solution),digits=6)) "*
            "muP_final=$(muPf)")
    outdir = joinpath(@__DIR__, "results", "trace"); mkpath(outdir)
    csv = joinpath(outdir, "$(NAME)__$(tag).csv")
    open(csv, "w") do io
        println(io, join(_COLS, ","))
        for row in rec.rows; println(io, join((row[c] for c in _COLS), ",")); end
    end
    println("  wrote $csv  ($(length(rec.rows)) iters)")
    finalize(nlp)
end

trace("superlin",  GateStatic(2.0, 2.0, 10.0, -1.0, 1.0, 1.0e7))   # E-gate kept, θ_P=2
trace("bcoupled",  BarrierCoupled(kappa_P=2.0, theta_P=1.0, muP_max=1.0e7))  # no E-gate, μ_B-coupled
