#=
    NCLModel.

    Adapted from the NCLModel in: https://github.com/JuliaSmoothOptimizers/NCL.jl/blob/main/src/NCLModel.jl

=#

@doc raw"""
    NCLModel

Reformulate the nonlinear program
```
min_x        f(x)
subject to   c♭ ≤ c(x) ≤ c♯
             x ≥ 0

```
using a (generalized-kernel) augmented-Lagrangian formulation. For a given scalar ``ρ``,
multiplier ``y_e`` and penalty kernel ``φ`` ([`AbstractKernel`](@ref)), the NCL model writes
```
min_{x,r}    f(x) - y_eᵀ r + \frac{1}{ρ} \sum_i φ(-ρ r_i)
subject to   c♭ ≤ c(x) + r ≤ c♯
             x ≥ 0

```
The quadratic kernel ``φ(t) = ½t²`` gives the penalty ``\tfrac{ρ}{2}\|r\|^2`` and recovers
MadNCL exactly. `scaled_arg` selects the kernel-argument convention (GeneralKernelKKT §9):
`true` (default) ⇒ argument ``-ρ r_i`` (the form above, convention **A**); `false` ⇒
``ρ`` pulled outside, penalty ``ρ \sum_i φ(r_i)`` (convention **B**). For the quadratic kernel
the two conventions coincide.
"""
struct NCLModel{T, VT, M, K} <: NLPModels.AbstractNLPModel{T, VT}
    nlp::M
    kernel::K
    scaled_arg::Bool
    nx::Int
    nr::Int
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::NLPModels.Counters
    yk::VT
    ρk::Base.RefValue{T}
end

function NCLModel(
    nlp::NLPModels.AbstractNLPModel{T, VT};
    resid::T=zero(T),
    ρ::T=one(T),
    kernel::AbstractKernel=QuadraticKernel(),
    scaled_arg::Bool=true,
) where {T, VT}

    nx = NLPModels.get_nvar(nlp)
    nr = NLPModels.get_ncon(nlp)
    nvar = nx + nr

    y = fill!(similar(VT, nr), one(T))

    lvarx = NLPModels.get_lvar(nlp)
    uvarx = NLPModels.get_uvar(nlp)

    lvarr = fill!(similar(VT, nr), -Inf)
    uvarr = fill!(similar(VT, nr),  Inf)

    x0 = NLPModels.get_x0(nlp)
    r0 = fill!(similar(VT, nr), resid)

    meta = NLPModels.NLPModelMeta{T, VT}(
        nvar;
        lvar=vcat(lvarx, lvarr),
        uvar=vcat(uvarx, uvarr),
        x0=vcat(x0, r0),
        y0=NLPModels.get_y0(nlp),
        nnzj=NLPModels.get_nnzj(nlp) + nr,
        nnzh=NLPModels.get_nnzh(nlp) + nr,
        ncon=nr,
        lcon=NLPModels.get_lcon(nlp),
        ucon=NLPModels.get_ucon(nlp),
        minimize=true,
    )
    cnt = NLPModels.Counters()

    return NCLModel{T, VT, typeof(nlp), typeof(kernel)}(
        nlp, kernel, scaled_arg, nx, nr, meta, cnt, y, Ref{T}(ρ),
    )
end

# Penalty objective term:  (1/ρ) Σᵢ φ(−ρ rᵢ)  [convention A]  or  ρ Σᵢ φ(rᵢ)  [convention B].
@inline function _penalty_obj(kernel::AbstractKernel, scaled_arg::Bool, ρ::T, r) where {T}
    if scaled_arg
        return mapreduce(ri -> phi(kernel, -ρ * ri), +, r; init=zero(T)) / ρ
    else
        return ρ * mapreduce(ri -> phi(kernel, ri), +, r; init=zero(T))
    end
end
# QuadraticKernel: exact closed form ⇒ bit-for-bit MadNCL (both conventions coincide here).
@inline _penalty_obj(::QuadraticKernel, ::Bool, ρ::T, r) where {T} = ρ * dot(r, r) / T(2)

set_barrier!(ncl::NCLModel, μ) = nothing

function NLPModels.obj(ncl::NCLModel{T, VT}, xr::VT) where {T, VT <: AbstractVector{T}}
    sense = NLPModels.get_minimize(ncl.nlp) ? T(1) : T(-1)
    x = view(xr, 1:ncl.nx)
    r = view(xr, 1+ncl.nx:ncl.nx+ncl.nr)
    obj_val = NLPModels.obj(ncl.nlp, x)
    obj_res = -dot(ncl.yk, r) + _penalty_obj(ncl.kernel, ncl.scaled_arg, ncl.ρk[], r)
    return sense * obj_val + obj_res
end

function NLPModels.cons!(ncl::NCLModel{T, VT}, xr::VT, cx::VT) where {T, VT <: AbstractVector{T}}
    x = view(xr, 1:ncl.nx)
    r = view(xr, 1+ncl.nx:ncl.nx+ncl.nr)
    NLPModels.cons!(ncl.nlp, x, cx)
    axpy!(one(T), r, cx)
    return cx
end

function NLPModels.grad!(ncl::NCLModel{T, VT}, xr::VT, gxr::VT) where {T, VT <: AbstractVector{T}}
    sense = NLPModels.get_minimize(ncl.nlp) ? T(1) : T(-1)
    x = view(xr, 1:ncl.nx)
    r = view(xr, 1+ncl.nx:ncl.nx+ncl.nr)
    gx = view(gxr, 1:ncl.nx)
    gr = view(gxr, 1+ncl.nx:ncl.nx+ncl.nr)
    NLPModels.grad!(ncl.nlp, x, gx)
    gx .*= sense
    ρ = ncl.ρk[]
    # Penalty gradient ∂/∂rᵢ:  −φ'(−ρ rᵢ)  [conv A]  or  ρ φ'(rᵢ)  [conv B].
    # For QuadraticKernel both reduce to −yk + ρ r, bit-for-bit (negation is exact).
    if ncl.scaled_arg
        gr .= .-ncl.yk .- dphi.(Ref(ncl.kernel), .-(ρ .* r))
    else
        gr .= .-ncl.yk .+ ρ .* dphi.(Ref(ncl.kernel), r)
    end
    return gxr
end

function NLPModels.jprod!(ncl::NCLModel{T, VT}, xr::VT, v::VT, Jv::VT) where {T, VT <: AbstractVector{T}}
    x = view(xr, 1:ncl.nx)
    vx = view(v, 1:ncl.nx)
    vr = view(v, 1+ncl.nx:ncl.nx+ncl.nr)
    NLPModels.jprod!(ncl.nlp, x, vx, Jv)
    axpy!(one(T), vr, Jv)
    return Jv
end

function NLPModels.jtprod!(ncl::NCLModel{T, VT}, xr::VT, v::VT, Jtv::VT) where {T, VT <: AbstractVector{T}}
    x = view(xr, 1:ncl.nx)
    Jtvx = view(Jtv, 1:ncl.nx)
    Jtvr = view(Jtv, 1+ncl.nx:ncl.nx+ncl.nr)
    NLPModels.jtprod!(ncl.nlp, x, v, Jtvx)
    Jtvr .= v
    return Jtv
end

function NLPModels.jac_structure!(ncl::NCLModel, jrows::VI, jcols::VI) where {VI <: AbstractVector{Int}}
    m, n = NLPModels.get_ncon(ncl), NLPModels.get_nvar(ncl)
    nnjx = NLPModels.get_nnzj(ncl.nlp)
    nnjxr = NLPModels.get_nnzj(ncl)
    jrowsx = view(jrows, 1:nnjx)
    jcolsx = view(jcols, 1:nnjx)
    NLPModels.jac_structure!(ncl.nlp, jrowsx, jcolsx)
    jrows[nnjx+1:nnjxr] .= 1:m
    jcols[nnjx+1:nnjxr] .= ncl.nx+1:n
    return (jrows, jcols)
end

function NLPModels.jac_coord!(ncl::NCLModel{T, VT}, xr::VT, jac::VT) where {T, VT <: AbstractVector{T}}
    nnjx = NLPModels.get_nnzj(ncl.nlp)
    nnjr = NLPModels.get_nnzj(ncl)
    x = view(xr, 1:ncl.nx)
    jacx = view(jac, 1:nnjx)
    NLPModels.jac_coord!(ncl.nlp, x, jacx)
    jac[nnjx+1:nnjr] .= one(T)
    return jac
end

function NLPModels.hess_structure!(ncl::NCLModel, hrows::VI, hcols::VI) where {VI <: AbstractVector{Int}}
    n = NLPModels.get_nvar(ncl)
    nnzhx = NLPModels.get_nnzh(ncl.nlp)
    nnzhxr = NLPModels.get_nnzh(ncl)
    hrowsx = view(hrows, 1:nnzhx)
    hcolsx = view(hcols, 1:nnzhx)
    NLPModels.hess_structure!(ncl.nlp, hrowsx, hcolsx)
    hrows[nnzhx+1:nnzhxr] .= ncl.nx+1:n
    hcols[nnzhx+1:nnzhxr] .= ncl.nx+1:n
    return (hrows, hcols)
end

function NLPModels.hess_coord!(
    ncl::NCLModel{T, VT},
    xr::AbstractVector{T},
    y::AbstractVector{T},
    hess::AbstractVector{T};
    obj_weight::T=one(T),
) where {T, VT <: AbstractVector{T}}
    sense = NLPModels.get_minimize(ncl.nlp) ? T(1) : T(-1)
    nnzhx = NLPModels.get_nnzh(ncl.nlp)
    nnzhxr = NLPModels.get_nnzh(ncl)
    x = view(xr, 1:ncl.nx)
    hessx = view(hess, 1:nnzhx)
    NLPModels.hess_coord!(ncl.nlp, x, y, hessx; obj_weight=sense * obj_weight)
    # Penalty Hessian diagonal = ρΣ, the *only* place the Σ generalization enters the linear
    # algebra: the KKT systems read this block as kkt.ρk and assemble −(1/ρ)Σ⁻¹ (K2r) /
    # ρAᵀΣA + Ω (K1s) elementwise from it, so they need no kernel-specific change.
    # Σᵢᵢ = φ''(arg): arg = −ρ rᵢ (conv A) or rᵢ (conv B). QuadraticKernel ⇒ ρ·1 = ρ (MadNCL).
    r = view(xr, ncl.nx+1:ncl.nx+ncl.nr)
    hpen = view(hess, nnzhx+1:nnzhxr)
    ρ = ncl.ρk[]
    if ncl.scaled_arg
        hpen .= ρ .* ddphi.(Ref(ncl.kernel), .-(ρ .* r))
    else
        hpen .= ρ .* ddphi.(Ref(ncl.kernel), r)
    end
    return hess
end

