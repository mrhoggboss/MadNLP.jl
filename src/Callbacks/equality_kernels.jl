# Penalty kernels and penalty-parameter schedules for `KernelPenaltyEquality`
# (see `src/Callbacks/nlpmodels.jl`).
#
# Each equality `c_E(x) = b` is reformulated with a free slack `s` (coupling
# `c_E(x) - s = b`) and a smooth penalty `φ(μ, s)` added to the objective. Driving
# `μ → ∞` forces `s → 0`, recovering `c_E(x) = b`. A kernel bundles the three scalar
# quantities MadNLP needs:
#   kernel_val(kernel, μ, s)  = φ(μ, s)         (added to the objective)
#   kernel_grad(kernel, μ, s) = ∂φ/∂s           (added to the slack gradient → KKT RHS)
#   kernel_hess(kernel, μ, s) = ∂²φ/∂s² (= D_s) (added to pr_diag → KKT (1,1) block)
#
# The scalar kernel functions are `@inline` and branchless so they broadcast on the GPU
# (the condensed/cuDSS path is a future stage).

"""
    AbstractEqualityKernel

Penalty kernel for [`KernelPenaltyEquality`](@ref). Concrete subtypes implement
`kernel_val`, `kernel_grad`, and `kernel_hess` at a scalar slack `s` and penalty
parameter `μ`.
"""
abstract type AbstractEqualityKernel end

# These kernels are deliberately NOT overflow-safe. cosh/sinh overflow `T` once
# |μ·s| ≳ log(floatmax(T)) (≈709 for Float64). When that happens the penalty value/grad/
# Hessian becomes non-finite and MadNLP terminates the solve with a clear message (see
# `_penalty_overflow` in `src/IPM/callbacks.jl`). We prefer to fail loudly: a saturated
# penalty is no longer the true penalty and would yield a wrong Newton step. Keep `μ` in a
# sane range (use a μ-continuation schedule) or pick the normalized/quadratic kernel.

"""
    QuadraticKernel <: AbstractEqualityKernel

Convex quadratic penalty `φ = (μ/2) s²` (gradient `μ s`, Hessian `μ`). The
baseline: `D_s = μ` is a positive constant, so the augmented system stays
nonsingular and the condensed system stays SPD.
"""
struct QuadraticKernel <: AbstractEqualityKernel end
@inline kernel_val( ::QuadraticKernel, μ::T, s::T) where {T} = (μ * s * s) / 2
@inline kernel_grad(::QuadraticKernel, μ::T, s::T) where {T} = μ * s
@inline kernel_hess(::QuadraticKernel, μ::T, s::T) where {T} = μ

"""
    CoshKernel <: AbstractEqualityKernel

Exponential penalty `φ = cosh(μ s)` (gradient `μ sinh(μ s)`, Hessian `μ² cosh(μ s)`).
`μ` scales the argument. Not overflow-safe: if `|μ s| ≳ 709` (Float64) the term overflows
and the solve is terminated with an error — keep `μ` moderate or use a continuation schedule.
"""
struct CoshKernel <: AbstractEqualityKernel end
@inline kernel_val( ::CoshKernel, μ::T, s::T) where {T} = cosh(μ * s)
@inline kernel_grad(::CoshKernel, μ::T, s::T) where {T} = μ * sinh(μ * s)
@inline kernel_hess(::CoshKernel, μ::T, s::T) where {T} = μ^2 * cosh(μ * s)

"""
    CoshNormalizedKernel <: AbstractEqualityKernel

μ-normalized exponential penalty `φ = cosh(μ s)/μ` (gradient `sinh(μ s)`, Hessian
`μ cosh(μ s)`) — the [`CoshKernel`](@ref) value/gradient/Hessian each divided by `μ`, hence
less overflow-prone, but still terminated (not saturated) on overflow. (The constant `1/μ`
offset of the Bertsekas form is dropped: it has no effect on the gradient, Hessian, or iterates.)
"""
struct CoshNormalizedKernel <: AbstractEqualityKernel end
@inline kernel_val( ::CoshNormalizedKernel, μ::T, s::T) where {T} = cosh(μ * s) / μ
@inline kernel_grad(::CoshNormalizedKernel, μ::T, s::T) where {T} = sinh(μ * s)
@inline kernel_hess(::CoshNormalizedKernel, μ::T, s::T) where {T} = μ * cosh(μ * s)

"""
    AbstractPenaltySchedule

Strategy for advancing the penalty parameter `μ` of a [`KernelPenaltyEquality`](@ref)
handler across interior-point iterations. Concrete subtypes implement
`update_penalty!(schedule, handler, solver)`, called once per iteration after the barrier
update. Add new schedules (static continuation, adaptive, …) by subtyping — no solver edits.
"""
abstract type AbstractPenaltySchedule end

"""
    FixedPenalty <: AbstractPenaltySchedule

Keep `μ` constant at its initial value (no continuation). The Stage-1 default.
"""
struct FixedPenalty <: AbstractPenaltySchedule end
# `update_penalty!(::FixedPenalty, handler, solver)` is a no-op; defined in
# `src/IPM/solver.jl` next to the solver-loop hook so `solver` accessors are in scope.

"""
    StaticContinuation(; kappa_P, theta_P, s_thresh, opt_tol_coef = 1.0, opt_tol_exp = -1.0, muP_max = Inf)
        <: AbstractPenaltySchedule

Static (fixed-parameter) μ_P continuation, gated on the subproblem (decoupled from μ_B). Once per
iteration, bump `μ_P` **iff both** gates hold:

  1. the penalty-barrier subproblem optimality error is small enough:
         `get_inf_barrier(solver) ≤ ε_P(μ_P)`,  with  `ε_P(μ_P) = opt_tol_coef · μ_P^opt_tol_exp`.
     The default `opt_tol_coef = 1, opt_tol_exp = -1` gives `ε_P(μ_P) = 1/μ_P`. Here
     `get_inf_barrier = max(inf_pr, inf_du, inf_compl_mu)`; `inf_du` already includes the penalty
     stationarity φ'(s) − y, so this is the *penalty*-barrier subproblem error.
  2. the equality-slack ∞-norm is small enough to keep the cosh argument well-conditioned:
         `‖s_E‖∞ ≤ s_thresh`   (μ_P·s is the cosh argument in `kernel_hess = μ_P²·cosh(μ_P·s)`).

The bump is the Ipopt-style superlinear step, capped:

    μ_P ← min(muP_max, max(kappa_P·μ_P, μ_P^theta_P))

`kappa_P > 1` (geometric) and `theta_P > 1` (superlinear; dominates once μ_P > 1). The framework
(`_update_penalty_mu!`) resets the filter and re-caches φ'(s)/the merit baseline whenever μ_P moves.
No overflow safeguard — if a bump drives μ_P·s past the cosh overflow threshold the solve
terminates with `InvalidNumberException` (signal to soften the gates / cap).

`n_bumps` and `muP_cur` are diagnostic refs (bump count and current μ_P, → final μ_P after the
solve), updated in place each iteration; construct a fresh schedule per solve to read them.
`update_penalty!(::StaticContinuation, handler, solver)` is defined in `src/IPM/solver.jl`.

Pass it to the treatment, e.g.
`KernelPenaltyEquality(CoshKernel(); schedule = StaticContinuation(kappa_P=10.0, theta_P=1.5, s_thresh=1e-2), muP=1.0)`.
"""
struct StaticContinuation <: AbstractPenaltySchedule
    kappa_P::Float64        # geometric growth factor κ_P (>1)
    theta_P::Float64        # superlinear growth exponent θ_P (>1)
    opt_tol_coef::Float64   # c in ε_P(μ_P) = c·μ_P^p  (optimality-error gate; default 1)
    opt_tol_exp::Float64    # p in ε_P(μ_P) = c·μ_P^p  (default -1 ⇒ ε_P = 1/μ_P)
    s_thresh::Float64       # τ_s : equality-slack ∞-norm gate (conditioning)
    muP_max::Float64        # hard cap on μ_P (Inf = uncapped)
    n_bumps::Base.RefValue{Int}      # diagnostic: # of μ_P bumps this solve
    muP_cur::Base.RefValue{Float64}  # diagnostic: current μ_P (→ final μ_P after the solve)
end
StaticContinuation(; kappa_P::Real, theta_P::Real, s_thresh::Real,
                   opt_tol_coef::Real = 1.0, opt_tol_exp::Real = -1.0, muP_max::Real = Inf) =
    StaticContinuation(Float64(kappa_P), Float64(theta_P), Float64(opt_tol_coef),
                       Float64(opt_tol_exp), Float64(s_thresh), Float64(muP_max), Ref(0), Ref(NaN))
