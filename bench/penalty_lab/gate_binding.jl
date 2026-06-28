# Probe: how often is the SLACK gate (‖s_E‖∞ ≤ s_thresh/μ_P) the binding constraint for the
# FIRST μ_P bump? The first bump is config-independent (gates + muP0 are shared across the grid),
# so one instrumented run characterizes the whole grid.
#
# StaticProbe mirrors StaticContinuation's gate logic and records, BEFORE the first bump:
#   n_s_block   : iters where opt-gate held but slack-gate failed  (slack gate SOLELY delaying)
#   n_opt_block : iters where slack-gate held but opt-gate failed
#   n_both_block: iters where both failed
# and at the first bump, which quantity is closest to its 1/μ_P threshold (the binding gate).
#
# Serial (1 core) over a size-stratified sample, to coexist with a running grid (≤24 cores).
# Usage: julia --project=bench/penalty bench/penalty_lab/gate_binding.jl [stride]
using MadNLP, CUTEst, LinearAlgebra, Printf

mutable struct StaticProbe <: MadNLP.AbstractPenaltySchedule
    kappa_P::Float64; theta_P::Float64; opt_tol_coef::Float64; opt_tol_exp::Float64
    s_thresh::Float64; muP_max::Float64
    n_s_block::Int; n_opt_block::Int; n_both_block::Int
    first_bump_iter::Int; binding::Symbol; bumped::Bool
end
StaticProbe(; kappa_P, theta_P, s_thresh=1.0, opt_tol_coef=1.0, opt_tol_exp=-1.0, muP_max=Inf) =
    StaticProbe(kappa_P, theta_P, opt_tol_coef, opt_tol_exp, s_thresh, muP_max, 0,0,0, -1, :none, false)

function MadNLP.update_penalty!(s::StaticProbe, eh::KernelPenaltyEquality, solver::MadNLP.AbstractMadNLPSolver{T}) where T
    μ = eh.muP[]
    μ >= s.muP_max && return
    E  = MadNLP.get_inf_barrier(solver)
    sE = isempty(eh.ind_eqslack) ? zero(T) :
         norm(view(MadNLP.slack(MadNLP.get_x(solver)), eh.ind_eqslack), Inf)
    otol = T(s.opt_tol_coef) * μ^T(s.opt_tol_exp)     # opt gate threshold ε_P(μ_P)
    stol = T(s.s_thresh) / μ                          # slack gate threshold s_thresh/μ_P
    opt_ok = E <= otol
    s_ok   = sE <= stol
    if opt_ok && s_ok
        if !s.bumped
            s.bumped = true
            s.first_bump_iter = MadNLP.get_cnt(solver).k
            s.binding = (sE/stol) >= (E/otol) ? :s : :opt   # tighter ratio ⇒ binding gate
        end
        eh.muP[] = min(T(s.muP_max), max(T(s.kappa_P)*μ, μ^T(s.theta_P)))
    elseif !s.bumped
        opt_ok && !s_ok ? (s.n_s_block += 1) :
        (!opt_ok && s_ok ? (s.n_opt_block += 1) : (s.n_both_block += 1))
    end
    return
end

stride = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
probs = String.(filter(!isempty, strip.(readlines(joinpath(@__DIR__,"regimes","le500_eq.txt")))))
sample = probs[1:stride:end]
println("probing $(length(sample)) of $(length(probs)) le500_eq problems (stride=$stride), serial...")

rows = NamedTuple[]
for name in sample
    local nlp
    try; nlp = CUTEstModel(name; decode=false); catch; continue; end
    sp = StaticProbe(kappa_P=2.0, theta_P=1.0)   # config-independent for the FIRST bump
    treat = KernelPenaltyEquality(CoshKernel(); schedule=sp, muP=1.0)
    st = "ERR"
    try
        r = madnlp(nlp; kkt_system=MadNLP.SparseCondensedKKTSystem, tol=1e-8, equality_treatment=treat,
                   max_iter=1000, max_wall_time=20.0, print_level=MadNLP.ERROR)
        st = string(r.status)
    catch; end
    finalize(nlp)
    push!(rows, (name=name, bumped=sp.bumped, fbi=sp.first_bump_iter, binding=sp.binding,
                 ns=sp.n_s_block, nopt=sp.n_opt_block, nboth=sp.n_both_block, status=st))
end

bumped = filter(r->r.bumped, rows)
nob = filter(r->!r.bumped, rows)
println("\n=== FIRST-μ_P-BUMP gate analysis (sample n=$(length(rows))) ===")
println("bumped at least once: $(length(bumped))   never bumped within budget: $(length(nob))")
if !isempty(bumped)
    sb = count(r->r.binding==:s, bumped); ob = count(r->r.binding==:opt, bumped)
    @printf("at the first bump, binding (tighter) gate:   slack=%d (%.0f%%)   opt=%d (%.0f%%)\n",
            sb, 100sb/length(bumped), ob, 100ob/length(bumped))
    s_delayed  = count(r->r.ns>0, bumped)
    o_delayed  = count(r->r.nopt>0, bumped)
    @printf("run-up to first bump: slack-gate SOLELY blocked ≥1 iter in %d/%d (%.0f%%) | opt-gate solely in %d/%d (%.0f%%)\n",
            s_delayed,length(bumped),100s_delayed/length(bumped), o_delayed,length(bumped),100o_delayed/length(bumped))
    med(f) = (v=sort([f(r) for r in bumped]); isempty(v) ? 0 : v[cld(length(v),2)])
    @printf("median iters solely-blocked: slack=%d  opt=%d  both=%d   median first_bump_iter=%d\n",
            med(r->r.ns), med(r->r.nopt), med(r->r.nboth), med(r->r.fbi))
end
println("\n  name              bumped fbi binding  s_block opt_block both_block  status")
for r in rows
    @printf("  %-16s  %-5s  %3d  %-5s   %5d   %6d    %6d    %s\n",
            r.name, r.bumped, r.fbi, r.binding, r.ns, r.nopt, r.nboth, r.status)
end
println("GATE_PROBE_DONE")
