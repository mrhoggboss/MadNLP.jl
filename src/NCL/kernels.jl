#=
    Penalty kernels for generalized-kernel NCL (the method of multipliers).

    A kernel bundles the scalar `φ` and its first two derivatives. The NCL penalty is built
    from `φ` evaluated at the (convention-dependent) kernel argument of the residual `r`:

      convention (A)  [default; `scaled_arg=true`]:
          penalty = (1/ρ) Σᵢ φ(−ρ rᵢ),   ∂/∂rᵢ = −φ'(−ρ rᵢ),   ∂²/∂rᵢ² = ρ φ''(−ρ rᵢ) = (ρΣ)ᵢᵢ
      convention (B)  [ρ pulled outside; `scaled_arg=false`]:
          penalty = ρ Σᵢ φ(rᵢ),          ∂/∂rᵢ =  ρ φ'(rᵢ),      ∂²/∂rᵢ² = ρ φ''(rᵢ)

    `Σ = diag(φ''(arg))` is the *only* object that changes between MadNCL (quadratic) and a
    general kernel; `φ'' > 0` preserves every structural property NCL relies on (SQD of K2r,
    SPD-after-regularization of K1s, the inertia equivalence, the sparsity pattern). These
    expose the RAW scalar `φ/φ'/φ''` (no μ-scaling). Where Σ lands in the KKT systems is
    documented in `src/NCL/k2r.jl` / `src/NCL/k1s.jl`.

    Ported from the standalone MadNCL package (`madncl_reference/src/Models/kernels.jl`).
    NOTE: in MadNLP's `KernelNCL` equality treatment the relaxation residual is stored as the
    free slack `s = c(x) − b = −r`, so the injected kernel argument is `+ρ s` (= −ρ r); see
    `src/IPM/callbacks.jl` and `src/IPM/kernels.jl`.
=#

"""
    AbstractKernel

Scalar penalty kernel for NCL. Concrete subtypes implement [`phi`](@ref), [`dphi`](@ref),
[`ddphi`](@ref) (= ``φ``, ``φ'``, ``φ''``) of a scalar argument. [`QuadraticKernel`](@ref)
(``φ = ½t²``) recovers MadNCL's quadratic augmented Lagrangian exactly.
"""
abstract type AbstractKernel end

"""
    QuadraticKernel <: AbstractKernel

``φ(t) = ½t²``, ``φ'(t) = t``, ``φ''(t) = 1`` ⟹ ``Σ = I``. Recovers classic quadratic NCL
exactly (bit-for-bit).
"""
struct QuadraticKernel <: AbstractKernel end
@inline phi(::QuadraticKernel, t::T) where {T} = t * t / T(2)
@inline dphi(::QuadraticKernel, t::T) where {T} = t
@inline ddphi(::QuadraticKernel, t::T) where {T} = one(T)

"""
    CoshKernel <: AbstractKernel

``φ(t) = cosh(t) − 1``, ``φ'(t) = sinh(t)``, ``φ''(t) = cosh(t)`` ⟹ ``Σ = diag(cosh(arg)) ⪰ I``.
Not overflow-safe: `cosh`/`sinh` overflow once `|arg| ≳ 709` (Float64).
"""
struct CoshKernel <: AbstractKernel end
@inline phi(::CoshKernel, t::T) where {T} = cosh(t) - one(T)
@inline dphi(::CoshKernel, t::T) where {T} = sinh(t)
@inline ddphi(::CoshKernel, t::T) where {T} = cosh(t)

# (φ')⁻¹ : inverse of the first derivative, for the multiplier-based residual predictor.
# At a subproblem stationary point (conv A) φ'(−ρr*) = y−yₖ ⟹ r* = −(φ')⁻¹(y−yₖ)/ρ.
@inline inv_dphi(::QuadraticKernel, v::T) where {T} = v          # φ'(t)=t
@inline inv_dphi(::CoshKernel, v::T) where {T} = asinh(v)        # φ'(t)=sinh t  (asinh tames the spike)
