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
    StaticContinuation(; kappa_P, theta_P, s_thresh = 1.0, opt_tol_coef = 1.0, opt_tol_exp = -1.0, muP_max = Inf)
        <: AbstractPenaltySchedule

Static (fixed-parameter) μ_P continuation, gated on the subproblem (decoupled from μ_B). Once per
iteration, bump `μ_P` **iff both** gates hold:

  1. the penalty-barrier subproblem optimality error is small enough:
         `get_inf_barrier(solver) ≤ ε_P(μ_P)`,  with  `ε_P(μ_P) = opt_tol_coef · μ_P^opt_tol_exp`.
     The default `opt_tol_coef = 1, opt_tol_exp = -1` gives `ε_P(μ_P) = 1/μ_P`. Here
     `get_inf_barrier = max(inf_pr, inf_du, inf_compl_mu)`; `inf_du` already includes the penalty
     stationarity φ'(s) − y, so this is the *penalty*-barrier subproblem error.
  2. the equality slack is small enough at the NEXT μ_P (the post-bump value `μ_P⁺ = min(muP_max,
     max(kappa_P·μ_P, μ_P^theta_P))`) to keep the cosh argument bounded:
         `‖s_E‖∞ ≤ s_thresh / μ_P⁺`   (so `μ_P⁺·‖s_E‖∞ ≤ s_thresh` — the cosh argument μ_P·s in
         `kernel_hess = μ_P²·cosh(μ_P·s)` stays ≲ s_thresh right AFTER the bump). Default `s_thresh = 1`.

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
`KernelPenaltyEquality(CoshKernel(); schedule = StaticContinuation(kappa_P=10.0, theta_P=1.5), muP=1.0)`.
"""
struct StaticContinuation <: AbstractPenaltySchedule
    kappa_P::Float64        # geometric growth factor κ_P (>1)
    theta_P::Float64        # superlinear growth exponent θ_P (>1)
    opt_tol_coef::Float64   # c in ε_P(μ_P) = c·μ_P^p  (optimality-error gate; default 1)
    opt_tol_exp::Float64    # p in ε_P(μ_P) = c·μ_P^p  (default -1 ⇒ ε_P = 1/μ_P)
    s_thresh::Float64       # τ_s : gate ‖s_E‖∞ ≤ s_thresh/μ_P (bounds cosh arg μ_P·s ≲ s_thresh); default 1
    muP_max::Float64        # hard cap on μ_P (Inf = uncapped)
    n_bumps::Base.RefValue{Int}      # diagnostic: # of μ_P bumps this solve
    muP_cur::Base.RefValue{Float64}  # diagnostic: current μ_P (→ final μ_P after the solve)
end
StaticContinuation(; kappa_P::Real, theta_P::Real, s_thresh::Real = 1.0,
                   opt_tol_coef::Real = 1.0, opt_tol_exp::Real = -1.0, muP_max::Real = Inf) =
    StaticContinuation(Float64(kappa_P), Float64(theta_P), Float64(opt_tol_coef),
                       Float64(opt_tol_exp), Float64(s_thresh), Float64(muP_max), Ref(0), Ref(NaN))

"""
    AugmentedLagrangian(; kappa_P=10.0, theta_P=1.0, eta_lambda=0.5, lambda_max=Inf,
                          muP_max=1.0e7, omega0=1.0, beta_omega=0.2, stall_patience=6,
                          bump_patience=5, opt_tol_coef=1.0, opt_tol_exp=-1.0)

Hestenes–Powell method of multipliers on the free equality slack `s = c_E(x) - b`, run as a proper
LANCELOT/Conn–Gould–Toint outer iteration (this is the corrected schedule that closes the
certification gap; see `update_penalty!(::AugmentedLagrangian, …)` in `src/IPM/solver.jl`).

An OUTER step is taken once the inner penalty-barrier subproblem is solved to the CURRENT inner
tolerance `ω_k` (`get_inf_barrier ≤ ω_k`) OR the inner solve has genuinely STALLED — i.e.
`get_inf_barrier` has not improved (by ≥10%) for `stall_patience` consecutive IPM iterations. The
stall net is essential and is a true no-progress test (NOT a fixed iteration count): when the
slack-row stationarity floors `inf_du` ABOVE `ω_k`, a pure `ω_k` gate deadlocks, so the stall net
forces the corrective multiplier update that breaks the floor — but only once the inner solve has
actually plateaued, so the forced step uses an `s` at its current-λ equilibrium (never a noisy
mid-descent `s`, which would spuriously inflate `μ_P`). At each outer step:

  1. Update `λ ← clip(λ + μ_P·s, ±lambda_max)` (skipped only when the step `‖μ_P·s‖∞ ≤ tol` is a
     no-op). Slack-row stationarity is `λ + μ_P·s = y`, so this drives `λ → y`, hence
     `s = (y−λ)/μ_P → 0` (exact equality feasibility) at FINITE `μ_P`. DECOUPLED from (2).
  2. Increase `μ_P ← min(muP_max, max(kappa_P·μ_P, μ_P^theta_P))` ONLY on a CONSERVATIVE feasibility
     stall: `‖s‖∞` failed to contract below `eta_lambda·(anchor ‖s‖∞)` for `bump_patience`
     consecutive λ-updates (the multiplier iteration genuinely cannot reach feasibility at this
     `μ_P`). NEVER an alternative to (1), and never on a single noisy step. This keeps `μ_P` moderate
     so `D_s = μ_P` (hence the condensed system and the inertia perturbation `del_w`) stays
     well-conditioned — the conditioning that, when `μ_P` ballooned, floored `inf_du` (mode A) and
     stalled the barrier (mode B).
  3. Tighten the inner tolerance `ω_k ← max(tol/10, beta_omega·ω_k)` so successive λ-updates drive
     the dual stationarity → tol (decoupled from `μ_P`).

The schedule freezes (no `φ_A` move, no filter reset) at the AL fixed point: `‖s‖∞ ≤ tol` AND either
`get_inf_du ≤ tol` (success imminent) or `‖μ_P·s‖∞ ≤ tol` (`λ` converged). This replaces the old
buggy `‖s‖∞ ≤ tol`-only freeze, which stranded `λ` short of `y`.

The Hessian `D_s = μ_P` is UNCHANGED (the `λ·s` term is linear), so the condensed system stays SPD.
For pure fixed-`μ_P` method of multipliers set `kappa_P = 1, muP_max = muP0`. `opt_tol_coef`/
`opt_tol_exp` are legacy (unused by the new gate) and kept only for constructor back-compat.
"""
struct AugmentedLagrangian <: AbstractPenaltySchedule
    kappa_P::Float64        # μ_P growth factor on a genuine feasibility stall
    theta_P::Float64        # superlinear μ_P growth exponent
    opt_tol_coef::Float64   # (legacy; unused by the new ω_k gate — kept for back-compat)
    opt_tol_exp::Float64    # (legacy; unused by the new ω_k gate — kept for back-compat)
    eta_lambda::Float64     # feasibility-contraction target: bump μ_P after bump_patience λ-updates fail to drop ‖s‖ below eta_lambda·anchor
    lambda_max::Float64     # clip on |λ| (Inf = unclipped)
    muP_max::Float64        # hard cap on μ_P
    omega0::Float64         # initial inner-optimality tolerance ω_0
    beta_omega::Float64     # geometric tightening factor for ω_k ∈ (0,1)
    stall_patience::Int     # force an outer step after this many inner iters with NO get_inf_barrier improvement
    bump_patience::Int      # grow μ_P only after this many consecutive λ-updates without sufficient ‖s‖ contraction
    s_ref::Base.RefValue{Float64}    # ‖s_E‖∞ at the current bump-window anchor (init Inf ⇒ first step never bumps μ_P)
    n_anchor::Base.RefValue{Int}     # n_updates at the current anchor (window length = n_updates - n_anchor)
    omega::Base.RefValue{Float64}    # current ω_k
    e_best::Base.RefValue{Float64}   # best (min) get_inf_barrier seen in the current inner solve
    stall_cnt::Base.RefValue{Int}    # consecutive inner iters with no get_inf_barrier improvement
    n_updates::Base.RefValue{Int}    # diagnostic: # λ updates
    n_bumps::Base.RefValue{Int}      # diagnostic: # μ_P bumps
    muP_cur::Base.RefValue{Float64}  # diagnostic: current μ_P (→ final μ_P after the solve)
end
AugmentedLagrangian(; kappa_P::Real = 10.0, theta_P::Real = 1.0, eta_lambda::Real = 0.5,
                    lambda_max::Real = Inf, muP_max::Real = 1.0e7, omega0::Real = 1.0,
                    beta_omega::Real = 0.2, stall_patience::Integer = 6, bump_patience::Integer = 5,
                    opt_tol_coef::Real = 1.0, opt_tol_exp::Real = -1.0) =
    AugmentedLagrangian(Float64(kappa_P), Float64(theta_P), Float64(opt_tol_coef), Float64(opt_tol_exp),
                        Float64(eta_lambda), Float64(lambda_max), Float64(muP_max), Float64(omega0),
                        Float64(beta_omega), Int(stall_patience), Int(bump_patience),
                        Ref(Inf), Ref(0), Ref(Float64(omega0)), Ref(Inf), Ref(0), Ref(0), Ref(0), Ref(NaN))
