# μ_P continuation lab — STRATEGY definitions.
# (Responsibility: define μ_P update strategies. This is the file I edit when you describe a
#  new strategy. Each strategy is a subtype of `MadNLP.AbstractPenaltySchedule` plus one
#  `MadNLP.update_penalty!` method whose ONLY job is to set `eh.muP[]`.)
#
# WHERE A STRATEGY ENTERS THE SOLVER (exact code locations):
#   • abstract type / shipped strategies .... src/Callbacks/equality_kernels.jl:76 (abstract),
#       FixedPenalty, StaticContinuation (subproblem-gated μ_P continuation)
#   • the handler carries the strategy ....... src/Callbacks/nlpmodels.jl:123 (field `schedule`)
#   • per-iteration hook in regular! ......... src/IPM/solver.jl:303  update_penalty_mu!(solver)
#       called right after update_barrier! (:300), before set_aug_diagonal! (:307)
#   • framework reaction to a μ_P move ....... src/IPM/solver.jl:227  _update_penalty_mu!:
#       records μ_P_old, calls update_penalty!(schedule, eh, solver), and on a CHANGE resets the
#       filter + refreshes slack(f)=φ'(s;μ_P_new) + the Armijo baseline. The strategy itself
#       never touches the filter/gradient.
#   • the extension point ..................... update_penalty!(schedule, eh, solver) — set eh.muP[].
#
# ACCESSORS available inside update_penalty!(s, eh, solver) (full freedom; μ_P is DECOUPLED
# from μ_B unless the strategy chooses otherwise):
#   eh.muP[]                         current μ_P            eh.muP0       initial μ_P
#   eh.ind_eqslack, eh.b             eq-slack indices / targets
#   MadNLP.get_mu(solver)            barrier μ_B           get_opt(solver).barrier.mu_init  μ_B0
#   MadNLP.get_cnt(solver).k         iteration count
#   MadNLP.get_inf_pr/ du/ compl(solver)   primal/dual/compl infeasibility (current iterate)
#   MadNLP.slack(MadNLP.get_x(solver))[eh.ind_eqslack]   the equality slacks s (= residual proxy)
#   MadNLP.get_obj_val(solver)       internal merit objective
# Assign the new value with `eh.muP[] = T(...)` where `T = typeof(eh.muP[])` for type stability.

using MadNLP, LinearAlgebra

# FixedPenalty and StaticContinuation(; kappa_P, theta_P, s_thresh, …) ship in MadNLP (exported).
# (StaticContinuation is the subproblem-gated μ_P continuation; bumps μ_P when get_inf_barrier ≤
#  ε_P(μ_P) and ‖s_E‖∞ ≤ s_thresh.) Add more strategies below.

# ── EXAMPLE TEMPLATE 1: power law in μ_B with a hard floor (decoupled-capable variant) ──
# μ_P = clamp( muP0 * (muB0/μ_B)^rho , muP0 , muP_max ). This is the μ_B-tracking continuation
# (the *previous* StaticContinuation); kept here as an editable template for parameter studies.
struct PowerLawMuB <: MadNLP.AbstractPenaltySchedule
    rho::Float64
    muP_max::Float64
end
PowerLawMuB(; rho = 1.0, muP_max = Inf) = PowerLawMuB(Float64(rho), Float64(muP_max))
function MadNLP.update_penalty!(s::PowerLawMuB, eh::KernelPenaltyEquality, solver::MadNLP.AbstractMadNLPSolver{T}) where {T}
    muB0 = T(MadNLP.get_opt(solver).barrier.mu_init)
    muB  = MadNLP.get_mu(solver)
    eh.muP[] = clamp(eh.muP0 * (muB0 / muB)^T(s.rho), eh.muP0, T(s.muP_max))
    return
end

# ── EXAMPLE TEMPLATE 2: geometric step every K iterations (fully decoupled from μ_B) ──
# μ_P *= factor every `every` iterations, capped at muP_max. Demonstrates a strategy that
# reads only the iteration counter.
struct GeometricEveryK <: MadNLP.AbstractPenaltySchedule
    factor::Float64
    every::Int
    muP_max::Float64
end
GeometricEveryK(; factor = 10.0, every = 5, muP_max = Inf) =
    GeometricEveryK(Float64(factor), Int(every), Float64(muP_max))
function MadNLP.update_penalty!(s::GeometricEveryK, eh::KernelPenaltyEquality, solver::MadNLP.AbstractMadNLPSolver{T}) where {T}
    k = MadNLP.get_cnt(solver).k
    (k > 0 && k % s.every == 0) && (eh.muP[] = min(T(s.muP_max), eh.muP[] * T(s.factor)))
    return
end

# (Add strategies you describe below, following the same pattern.)
