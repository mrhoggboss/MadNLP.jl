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
