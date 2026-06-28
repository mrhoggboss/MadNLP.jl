# Penalty contributions for the `KernelPenaltyEquality` treatment. Both dispatch to a
# no-op for any other treatment, so non-penalty solves are untouched. The penalty acts on
# the equality slacks `view(s, ind_eqslack)`; broadcasts/reductions over that view are
# GPU-safe (no scalar indexing). See `src/Callbacks/equality_kernels.jl`.
#
# The kernels are NOT overflow-safe: if `cosh(μ s)` overflows, the penalty term is
# non-finite and we terminate the solve loudly (rather than silently saturating, which
# would corrupt the Newton step). `_penalty_overflow` logs a clear, actionable message and
# throws `InvalidNumberException`, which `solve!` turns into an `INVALID_NUMBER_*` status.
@noinline function _penalty_overflow(solver, eh::KernelPenaltyEquality, sym::Symbol)
    @error(get_logger(solver), string(
        "KernelPenaltyEquality: penalty kernel ", nameof(typeof(eh.kernel)),
        " produced a non-finite ", sym, " term (μ_P = ", eh.muP[],
        "). The penalty is too stiff at the current iterate — reduce muP, apply a ",
        "μ-continuation schedule, or use CoshNormalizedKernel / QuadraticKernel."))
    throw(InvalidNumberException(sym))
end

# Pure equality-penalty sum `Σ φ(μ, sᵢ)` over the current slacks; `zero` for any non-penalty
# treatment. No overflow check — used both inside eval_f_wrapper and when stripping the
# penalty back out to report the true objective.
penalty_objective(::AbstractEqualityTreatment, s) = zero(eltype(s))
function penalty_objective(eh::KernelPenaltyEquality, s)
    μ = eh.muP[]
    kernel = eh.kernel        # bind the singleton kernel locally: the mapreduce closure must
    se = view(s, eh.ind_eqslack)   # capture only bitstype values (kernel, μ) — NOT the handler
    pen = mapreduce(si -> kernel_val(kernel, μ, si), +, se; init = zero(eltype(s)))   # (eh holds GPU arrays)
    return pen + dot(eh.lambda, se)   # φ_A = Σφ(μ,sᵢ) + λᵀs (augmented-Lagrangian linear term; λ≡0 ⇒ pure penalty)
end

# Checked version used by eval_f_wrapper: terminate the solve if the penalty overflowed.
function penalty_objective(solver, eh::AbstractEqualityTreatment, s)
    pen = penalty_objective(eh, s)
    is_valid(pen) || _penalty_overflow(solver, eh, :obj)
    return pen
end

# `get_obj_val(solver)` carries the equality penalty folded in by eval_f_wrapper. This
# strips it back out, recovering the true model objective `f(x)` in MadNLP's internal
# (scaled, sign-flipped) space — what should be *reported* to the user. The penalty stays in
# the merit/filter (via `get_obj_val`). No-op (subtracts 0) for every non-penalty treatment.
true_obj_val(solver) = get_obj_val(solver) - penalty_objective(get_cb(solver).equality_handler, slack(get_x(solver)))

penalty_gradient!(solver, ::AbstractEqualityTreatment, sf, s) = nothing
function penalty_gradient!(solver, eh::KernelPenaltyEquality, sf, s)
    μ = eh.muP[]
    es = eh.ind_eqslack
    # Overwrite (not accumulate): `slack(f)` persists across iterations and is otherwise 0,
    # so only the equality-slack entries carry φ'_A(s); the rest must stay 0.
    @views sf[es] .= kernel_grad.(Ref(eh.kernel), μ, s[es]) .+ eh.lambda   # φ'_A = φ'(μ,s) + λ
    is_valid(view(sf, es)) || _penalty_overflow(solver, eh, :grad)
    return
end

# True equality-feasibility gate for termination: ‖s_E‖∞ ≤ tol. The freed slack s = c_E(x)-b is NOT
# part of get_inf_total, so a pure/augmented penalty can satisfy the scaled KKT with s still loose;
# this conjunct keeps SOLVE_SUCCEEDED honest (and is what lets the ALM certify s→0). No-op (always
# feasible) for every non-penalty treatment.
penalty_eq_feasible(::AbstractEqualityTreatment, solver, tol) = true
function penalty_eq_feasible(eh::KernelPenaltyEquality, solver, tol)
    isempty(eh.ind_eqslack) && return true
    return norm(view(slack(get_x(solver)), eh.ind_eqslack), Inf) <= tol
end

function eval_f_wrapper(solver::AbstractMadNLPSolver{T}, x::PrimalVector{T}) where T
    nlp = get_nlp(solver)
    cnt = get_cnt(solver)
    @trace(get_logger(solver),"Evaluating objective.")
    cnt.eval_function_time += @elapsed begin
        # NOTE: we flip the objective here so the rest of MadNLP internals
        # are the same, between min and max. when showing the objective value
        # to the user (in MadNLPExecutionStats) we flip it back (#517).
        sense = (get_minimize(nlp) ? one(T) : -one(T))
        obj_val = sense * _eval_f_wrapper(get_cb(solver), variable(x))
        # Add the equality penalty in the (minimized) internal objective space (#517),
        # so the filter / merit see it.
        obj_val += penalty_objective(solver, get_cb(solver).equality_handler, slack(x))
    end
    cnt.obj_cnt += 1
    if cnt.obj_cnt == 1 && !is_valid(obj_val)
        throw(InvalidNumberException(:obj))
    end
    return obj_val
end

function eval_grad_f_wrapper!(solver::AbstractMadNLPSolver, f::PrimalVector{T}, x::PrimalVector{T}) where T
    nlp = get_nlp(solver)
    cnt = get_cnt(solver)
    @trace(get_logger(solver),"Evaluating objective gradient.")
    cnt.eval_function_time += @elapsed _eval_grad_f_wrapper!(
        get_cb(solver),
        variable(x),
        variable(f),
    )
    if !get_minimize(nlp)
        variable(f) .*= -one(T)
    end
    # Penalty gradient on the equality slacks → KKT RHS and dual-infeasibility (both read
    # full(f)). Stationarity on those rows becomes φ'(s) = y.
    penalty_gradient!(solver, get_cb(solver).equality_handler, slack(f), slack(x))
    cnt.obj_grad_cnt+=1

    if cnt.obj_grad_cnt == 1 && !is_valid(full(f))
        throw(InvalidNumberException(:grad))
    end
    return f
end

function eval_cons_wrapper!(solver::AbstractMadNLPSolver, c::AbstractVector{T}, x::PrimalVector{T}) where T
    nlp = get_nlp(solver)
    cnt = get_cnt(solver)
    @trace(get_logger(solver), "Evaluating constraints.")
    cnt.eval_function_time += @elapsed _eval_cons_wrapper!(
        get_cb(solver),
        variable(x),
        c,
    )
    view(c,get_ind_ineq(solver)) .-= slack(x)
    c .-= get_rhs(solver)
    cnt.con_cnt+=1
    if cnt.con_cnt == 1 && !is_valid(c)
        throw(InvalidNumberException(:cons))
    end
    return c
end

function eval_jac_wrapper!(solver::AbstractMadNLPSolver, kkt::AbstractKKTSystem, x::PrimalVector{T}) where T
    nlp = get_nlp(solver)
    cnt = get_cnt(solver)
    ns = length(get_ind_ineq(solver))
    @trace(get_logger(solver), "Evaluating constraint Jacobian.")
    jac = get_jacobian(kkt)
    cnt.eval_function_time += @elapsed _eval_jac_wrapper!(
        get_cb(solver),
        variable(x),
        jac,
        )
    compress_jacobian!(kkt)
    cnt.con_jac_cnt += 1
    if cnt.con_jac_cnt == 1 && !is_valid(jac)
        throw(InvalidNumberException(:jac))
    end
    @trace(get_logger(solver),"Constraint jacobian evaluation started.")
    return jac
end

function eval_lag_hess_wrapper!(solver::AbstractMadNLPSolver, kkt::AbstractKKTSystem, x::PrimalVector{T},l::AbstractVector{T};is_resto=false) where T
    nlp = get_nlp(solver)
    cnt = get_cnt(solver)
    @trace(get_logger(solver),"Evaluating Lagrangian Hessian.")
    hess = get_hessian(kkt)
    scale = (get_minimize(nlp) ? one(T) : -one(T)) * (is_resto ? zero(T) : one(T))
    cnt.eval_function_time += @elapsed _eval_lag_hess_wrapper!(
        get_cb(solver),
        variable(x),
        l,
        hess;
        obj_weight = scale,
    )
    compress_hessian!(kkt)
    cnt.lag_hess_cnt += 1
    if cnt.lag_hess_cnt == 1 && !is_valid(hess)
        throw(InvalidNumberException(:hess))
    end
    return hess
end

function eval_jac_wrapper!(solver::AbstractMadNLPSolver, kkt::AbstractDenseKKTSystem, x::PrimalVector{T}) where T
    nlp = get_nlp(solver)
    cnt = get_cnt(solver)
    ns = length(get_ind_ineq(solver))
    @trace(get_logger(solver), "Evaluating constraint Jacobian.")
    jac = get_jacobian(kkt)
    cnt.eval_function_time += @elapsed _eval_jac_wrapper!(
        get_cb(solver),
        variable(x),
        jac,
    )
    compress_jacobian!(kkt)
    cnt.con_jac_cnt+=1
    if cnt.con_jac_cnt == 1 && !is_valid(jac)
        throw(InvalidNumberException(:jac))
    end
    @trace(get_logger(solver),"Constraint jacobian evaluation started.")
    return jac
end

function eval_lag_hess_wrapper!(
    solver::AbstractMadNLPSolver,
    kkt::AbstractDenseKKTSystem{T, VT, MT, QN},
    x::PrimalVector{T},
    l::AbstractVector{T};
    is_resto=false,
) where {T, VT, MT, QN<:ExactHessian}
    nlp = get_nlp(solver)
    cnt = get_cnt(solver)
    @trace(get_logger(solver),"Evaluating Lagrangian Hessian.")
    hess = get_hessian(kkt)
    scale = is_resto ? zero(T) : get_minimize(nlp) ? one(T) : -one(T)
    cnt.eval_function_time += @elapsed _eval_lag_hess_wrapper!(
        get_cb(solver),
        variable(x),
        l,
        hess;
        obj_weight = scale,
    )
    compress_hessian!(kkt)
    cnt.lag_hess_cnt+=1
    if cnt.lag_hess_cnt == 1 && !is_valid(hess)
        throw(InvalidNumberException(:hess))
    end
    return hess
end

function eval_lag_hess_wrapper!(
    solver::AbstractMadNLPSolver,
    kkt::AbstractKKTSystem{T, VT, MT, QN},
    x::PrimalVector{T},
    l::AbstractVector{T};
    is_resto=false,
) where {T, VT, MT<:AbstractMatrix{T}, QN<:AbstractQuasiNewton{T, VT}}
    cb = get_cb(solver)
    cnt = get_cnt(solver)
    @trace(get_logger(solver), "Update BFGS matrices.")

    qn = kkt.quasi_newton
    Bk = kkt.hess
    sk, yk = qn.sk, qn.yk
    n = length(qn.sk)
    m = size(kkt.jac, 1)

    if cnt.obj_grad_cnt >= 2
        # Build sk = x+ - x
        copyto!(sk, 1, variable(get_x(solver)), 1, n)   # sₖ = x₊
        axpy!(-one(T), qn.last_x, sk)              # sₖ = x₊ - x
        # Build yk = ∇L+ - ∇L
        copyto!(yk, 1, variable(get_f(solver)), 1, n)   # yₖ = ∇f₊
        axpy!(-one(T), qn.last_g, yk)              # yₖ = ∇f₊ - ∇f
        if m > 0
            jtprod!(get_jacl(solver), kkt, l)
            yk .+= @view(get_jacl(solver)[1:n])         # yₖ += J₊ᵀ l₊
            _eval_jtprod_wrapper!(cb, qn.last_x, l, qn.last_jv)
            axpy!(-one(T), qn.last_jv, yk)         # yₖ += J₊ᵀ l₊ - Jᵀ l₊
        end
        # Update quasi-Newton approximation.
        update!(qn, Bk, sk, yk)
    else
        # Init quasi-Newton approximation
        g0 = variable(get_f(solver))
        f0 = get_obj_val(solver)
        init!(qn, Bk, g0, f0)
    end

    # Backup data for next step
    copyto!(qn.last_x, 1, variable(get_x(solver)), 1, n)
    copyto!(qn.last_g, 1, variable(get_f(solver)), 1, n)

    compress_hessian!(kkt)
    return get_hessian(kkt)
end

