# ALM Phase-0 GO/NO-GO diagnostic. For each problem (default: the 62 bcoup_k2 succeeded-not-tight),
# re-solve with the quad bcoup_k2 config and capture at the FINAL iterate: the SCALED equality slack
# ‖s_E‖∞ (what the penalty/ALM act on), the equality multiplier ‖y_E‖∞, μ_P, and the UNSCALED
# eqfeas. Tests the multiplier-deficit premise (scaled s loose & y moderate ⇒ ALM tightens at finite
# μ_P) vs a scaling artifact (scaled s already tight, only unscaled loose ⇒ ALM can't help).
# Usage: julia --project=bench/penalty bench/penalty_lab/trace_alm_phase0.jl [namesfile]

using MadNLP, CUTEst, NLPModels, LinearAlgebra, Printf
const NAMESFILE = length(ARGS) >= 1 ? ARGS[1] :
    "/tmp/claude-1025/-home-yifan-xu-MadNLP-jl/76efbc10-da4a-42b3-ae87-d69db7eaec00/scratchpad/bucket62.txt"
const NAMES = String.(filter(!isempty, strip.(readlines(NAMESFILE))))

mutable struct BCoupled <: MadNLP.AbstractPenaltySchedule
    kappa_P::Float64; theta_P::Float64; muP_max::Float64
    last_muB::Base.RefValue{Float64}
end
BCoupled(; kappa_P, theta_P=1.0, muP_max=1.0e7) = BCoupled(kappa_P, theta_P, muP_max, Ref(NaN))
function MadNLP.update_penalty!(s::BCoupled, eh::KernelPenaltyEquality, solver::MadNLP.AbstractMadNLPSolver{T}) where T
    μB = Float64(MadNLP.get_mu(solver))
    if !isnan(s.last_muB[]) && μB < s.last_muB[] && eh.muP[] < s.muP_max
        eh.muP[] = min(T(s.muP_max), max(T(s.kappa_P)*eh.muP[], eh.muP[]^T(s.theta_P)))
    end
    s.last_muB[] = μB
    return
end

# recorder capturing the LAST-iterate penalty diagnostics
mutable struct P0 <: MadNLP.AbstractUserCallback
    s_inf::Float64; y_inf::Float64; muP::Float64
end
P0() = P0(NaN,NaN,NaN)
function (r::P0)(solver, mode)
    eh = MadNLP.get_cb(solver).equality_handler
    es = eh.ind_eqslack
    isempty(es) && return true
    r.s_inf = Float64(norm(view(MadNLP.slack(MadNLP.get_x(solver)), es), Inf))   # SCALED eq slack
    r.y_inf = Float64(norm(view(MadNLP.get_y(solver), es), Inf))                 # eq multiplier (scaled space)
    r.muP   = Float64(eh.muP[])
    return true
end

function _eqfeas(nlp, x)
    m = NLPModels.get_ncon(nlp); m == 0 && return 0.0
    c = similar(x, m); NLPModels.cons!(nlp, x, c)
    lc = NLPModels.get_lcon(nlp); uc = NLPModels.get_ucon(nlp)
    return Float64(maximum(abs.(c .- lc) .* (lc .== uc)))
end

@printf("%-12s %8s %12s %12s %12s %10s %9s %9s\n",
        "name","muP","scaled_s","unscaled_ef","y_E","s/ef","y/(muP*s)","status")
rows = NamedTuple[]
for name in NAMES
    nlp = try CUTEstModel(name; decode=false) catch; continue end
    rec = P0()
    sched = BCoupled(kappa_P=2.0, theta_P=1.0)
    treat = KernelPenaltyEquality(QuadraticKernel(); schedule=sched, muP=2.0)
    local r
    try
        r = madnlp(nlp; kkt_system=MadNLP.SparseCondensedKKTSystem, tol=1e-8, acceptable_tol=1e-8,
                   equality_treatment=treat, intermediate_callback=rec, max_iter=3000,
                   max_wall_time=120.0, print_level=MadNLP.ERROR)
    catch e; finalize(nlp); continue; end
    ef = _eqfeas(nlp, r.solution)
    s_over_ef = rec.s_inf / max(ef, 1e-300)            # ≈1 ⇒ con_scale≈1 (no scaling artifact)
    y_over_mus = rec.y_inf / max(rec.muP*rec.s_inf, 1e-300)  # ≈1 ⇒ stationarity y=μ_P·s held
    @printf("%-12s %8.0f %12.2e %12.2e %12.2e %10.2f %9.2f %s\n",
            name, rec.muP, rec.s_inf, ef, rec.y_inf, s_over_ef, y_over_mus, string(r.status))
    push!(rows, (name=name, muP=rec.muP, s=rec.s_inf, ef=ef, y=rec.y_inf, sef=s_over_ef))
    finalize(nlp)
end
# verdict
scaled_loose = count(r -> r.s > 1e-7, rows)
no_scale_art = count(r -> 0.1 <= r.sef <= 10, rows)        # scaled ≈ unscaled
y_moderate   = count(r -> r.y <= 1e3, rows)
@printf("\nPHASE0 over %d solved: scaled_s loose(>1e-7)=%d  no-scaling-artifact(0.1≤s/ef≤10)=%d  y_moderate(≤1e3)=%d\n",
        length(rows), scaled_loose, no_scale_art, y_moderate)
addressable = count(r -> r.s > 1e-7 && 0.1 <= r.sef <= 10 && r.y <= 1e3, rows)
@printf("  ⇒ ALM-addressable (scaled-loose ∧ no-scaling-artifact ∧ y-moderate): %d / %d\n", addressable, length(rows))
println("PHASE0_DONE")
