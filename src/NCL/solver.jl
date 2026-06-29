

#=
    NCLOptions
=#

@kwdef struct NCLOptions{T}
    verbose::Bool = true
    extrapolation::Bool = true
    extrapolation_rho::T = T(0.3)
    scaling::Bool = true
    scaling_max_gradient::T = T(1)
    opt_tol::T = T(1e-6)
    feas_tol::T = T(1e-6)
    constr_viol_tol::T = max(T(100) * feas_tol, T(1e-4))
    rho_init::T = T(1e2)
    rho_max::T = T(1e12)
    tau_rho::T = T(10)        # ρ growth factor in penalty continuation (branch c): ρ ← min(τ_ρ·ρ, ρ_max)
    predict_resid::Bool = false  # on a ρ-bump, reset the residual r (continuation predictor; avoids the cosh spike)
    predict_mode::Symbol = :geometric  # :geometric → r ← r·(ρ_prev/ρ_new);  :multiplier → r* = −(φ')⁻¹(y−yₖ)/ρ
    predict_skip_below::T = T(0)   # 3(a): skip the residual reset once pr_feas ≤ this (let the inner solve finish)
    predict_floor::T = T(0)        # 3(b): clamp |rᵢ| ≥ this after the reset (no over-shoot below the feasibility scale)
    predict_skip_dual::T = T(Inf)  # 3(a)': ALSO require ‖y−yₖ‖∞ ≤ this to skip (Inf ⇒ plain 3(a); protects the large-Δλ niche)
    predict_extrap::Bool = false   # idea (a): on the EXTRAPOLATION ρ-bump, also reset r ← r·(ρ_prev/ρ_new) so ρr stays
                                   # bounded through the extrapolation phase (the proven predictor move). Default off ⇒ unchanged.
    max_auglag_iter::Int = 20
    mu_init::T = T(1e-1)
    mu_tau::T = T(1.99)
    mu_fac::T = T(0.2)
    mu_min::T = T(1e-9)
    eta_mult::T = T(1)  # loosen the μ-reduce feasibility gate: reduce μ when pr_feas ≤ max(eta_mult·η, feas_tol).
                        # >1 lets μ advance while feasibility lags (safe for cosh: bounded r re-tightens). 1 ⇒ unchanged.
    eps_d_mult::T = T(1)  # scale the inner dual tolerance eps_d = eps_d_mult·100·μ^(1+γ). >1 ⇒ looser/cheaper
                          # subproblems (fewer inner Newton steps); <1 ⇒ tighter. 1 ⇒ unchanged.
    eps_d_cap::T = T(Inf)  # cap the inner tol: eps_d = min(eps_d_mult·100·μ^(1+γ), eps_d_cap). Bounds inf_pr =
                           # ‖c(x)−r‖∞ — the 2nd lever in the triangle bound ‖c(x)‖ ≤ inf_pr + ‖r‖. Inf ⇒ unchanged.
    tighten_success::Bool = false  # gate SUCCEEDED on the recomputed UNSCALED ‖c_E(x)−b‖∞ ≤ feas_tol (the TIGHT
                                   # metric), not the r-proxy. Pair with eps_d_cap so the inner solve can drive it down.
end

#=
    NCLSolver
=#

struct NCLSolver{T, VT, M, K}
    ncl::NCLModel{T, VT, M, K}
    ipm::MadNLPSolver{T, VT}
    options::NCLOptions{T}
    n::Int
    m::Int
end

function NCLSolver(nlp::NLPModels.AbstractNLPModel{T, VT}; ncl_options=NCLOptions{T}(),
                   kernel::AbstractKernel=QuadraticKernel(), scaled_arg::Bool=true,
                   ipm_options...) where {T, VT}
    ncl = if ncl_options.scaling
        NCLModel(ScaledModel(nlp; max_gradient=ncl_options.scaling_max_gradient);
                 kernel=kernel, scaled_arg=scaled_arg)
    else
        NCLModel(nlp; kernel=kernel, scaled_arg=scaled_arg)
    end
    solver = MadNLPSolver(
        ncl;
        nlp_scaling=false,
        ipm_options...,
    )
    m = n_constraints(solver.cb)
    # Remove regularization variables to work in the original space
    n = n_variables(solver.cb) - m
    @assert solver.m == m
    return NCLSolver{T, VT, typeof(ncl.nlp), typeof(ncl.kernel)}(ncl, solver, ncl_options, n, m)
end

#=
    NCLStats
=#

mutable struct NCLStats{T, VT} <: AbstractExecutionStats
    status::Status
    solution::VT
    regularization::VT
    objective::T
    dual_feas::T
    primal_feas::T
    multipliers::VT
    multipliers_L::VT
    multipliers_U::VT
    iter::Int
    counters::MadNLPCounters
end

function NCLStats(solver::NCLSolver{T, VT, M}, status) where {T, VT, M<:NLPModels.AbstractNLPModel}
    ipm = solver.ipm # MadNLP instance
    ncl = solver.ncl
    # Get original number of variables (before removing the fixed variables)
    n = NLPModels.get_nvar(ncl.nlp)
    m = solver.m
    is_min = NLPModels.get_minimize(ncl.nlp)
    ρk = ncl.ρk[]
    x = similar(VT, n + m)
    zl = similar(VT, n + m)
    zu = similar(VT, n + m)
    unpack_x!(x, ipm.cb, variable(solver.ipm.x))
    unpack_z!(zl, ipm.cb, variable(solver.ipm.zl))
    unpack_z!(zu, ipm.cb, variable(solver.ipm.zu))

    y = copy(solver.ipm.y)
    r = x[n+1:n+m]
    # Recover f(x) by removing the (kernel-dependent) penalty term added in NCLModel.obj.
    # _penalty_obj(::QuadraticKernel) = ρ·‖r‖²/2, so this is bit-identical to MadNCL for the quadratic.
    obj_val = solver.ipm.obj_val + dot(y, r) - _penalty_obj(ncl.kernel, ncl.scaled_arg, ρk, r)
    update_z!(ipm.cb, x, y, zl, zu, ipm.jacl)
    # Scale back problem data
    if isa(ncl.nlp, ScaledModel)
        obj_scale, con_scale = ncl.nlp.scaling_obj, ncl.nlp.scaling_cons
        obj_val /= obj_scale
        y .= y .* con_scale ./ obj_scale
        zl .= zl ./ obj_scale
        zu .= zu ./ obj_scale
    end
    return NCLStats{T, VT}(
        status,
        x[1:n],
        r,
        is_min ? obj_val : -obj_val,
        solver.ipm.inf_du,
        norm(r, Inf),
        is_min ? y : .-y,
        zl[1:n],
        zu[1:n],
        solver.ipm.cnt.k,
        solver.ipm.cnt,
    )
end

function getStatus(result::NCLStats)
    if result.status == SOLVE_SUCCEEDED
        println("Optimal solution found.")
    elseif result.status == INFEASIBLE_PROBLEM_DETECTED
        println("Convergence to an infeasible point.")
    elseif result.status == MAXIMUM_ITERATIONS_EXCEEDED
        println("Maximum number of iterations reached.")
    else
        println("Unknown return status.")
    end
end

#=
    NCL Algorithm
=#

function _introduce(nx, nr)
    println("MadNCL algorithm\n")

    println("Total number of variables............................:      ", nx)
    println("Total number of constraints..........................:      ", nr)
    println()
end

function _log_header()
    @printf(
        "outer  inner     objective    inf_pr   inf_du    η        μ       ρ \n"
    )
end

function _log_iter(nit, flag, n_inner, obj, inf_pr, inf_du, alpha, mu, rho)
    @printf(
        "%5s%1s %5i %+13.7e %6.2e %6.2e %6.2e %6.1e %6.2e\n",
        nit, flag, n_inner, obj, inf_pr, inf_du, alpha, mu, rho,
    )
end

function get_inf_du(solver::NCLSolver)
    return solver.ipm.inf_du
end

function get_constr_viol_tol(ncl::NCLModel, r::AbstractVector)
    return norm(r, Inf)
end

# Recomputed UNSCALED equality feasibility ‖c_E(x)−b‖∞ (the TIGHT metric) — to gate SUCCEEDED on the true
# constraint, not the r-proxy. For ScaledModel, cons!/lcon are scaled by con_scale, so divide it back out per row.
function _unscaled_eqfeas(ncl::NCLModel, x::AbstractVector)
    nlp = ncl.nlp
    m = NLPModels.get_ncon(nlp)
    m == 0 && return zero(eltype(x))
    c = similar(x, m); NLPModels.cons!(nlp, x, c)
    lc = NLPModels.get_lcon(nlp); uc = NLPModels.get_ucon(nlp)
    cs = hasproperty(nlp, :scaling_cons) ? nlp.scaling_cons : nothing
    e = zero(eltype(x))
    @inbounds for i in 1:m
        if lc[i] == uc[i]
            ri = abs(c[i] - lc[i])
            cs !== nothing && (ri /= cs[i])
            e = max(e, ri)
        end
    end
    return e
end
function get_constr_viol_tol(ncl::NCLModel{T, VT, M}, r::AbstractVector) where {T, VT, M<:ScaledModel}
    m = NLPModels.get_ncon(ncl)
    if m > 0
        con_scale = ncl.nlp.scaling_cons
        # Compute norm-Inf with mapreduce
        return mapreduce((x, c) -> abs(x) / c, max, r, con_scale)
    else
        return 0.0
    end
end

function setup!(solver::MadNLPSolver{T}; μ=1e-1, tol=1e-8) where T
    barrier = solver.opt.barrier
    # Update options
    barrier.mu_init = μ
    solver.opt.tol = tol
    solver.mu = barrier.mu_init
    # Ensure the barrier parameter is fixed
    barrier.mu_min = barrier.mu_init

    # Refresh values
    solver.obj_val = eval_f_wrapper(solver, solver.x)
    eval_grad_f_wrapper!(solver, solver.f, solver.x)
    eval_cons_wrapper!(solver, solver.c, solver.x)

    # Update filter
    theta = get_theta(solver.c)
    solver.theta_max = T(1e4) * max(1,theta)
    solver.theta_min = T(1e-4) * max(1,theta)
    solver.tau = max(solver.opt.tau_min,one(T)-barrier.mu_init)
    empty!(solver.filter)
    push!(solver.filter, (solver.theta_max,-Inf))

    return REGULAR
end

# RHS for extrapolation step
function set_aug_rhs_extrapolation!(solver::NCLSolver, kkt::AbstractKKTSystem, μ)
    ipm = solver.ipm
    ρ = solver.ncl.ρk[]

    n_ineq = length(ipm.ind_ineq)
    n, m = solver.n, solver.m

    f = primal(ipm.f)
    zl = full(ipm.zl)
    zu = full(ipm.zu)

    # Variables
    px = @view primal(ipm.p)[1:n]
    fx = @view primal(ipm.f)[1:n]
    jaclx = @view ipm.jacl[1:n]
    zlx = @view primal(ipm.zl)[1:n]
    zux = @view primal(ipm.zu)[1:n]
    # Slacks
    ps = @view primal(ipm.p)[n+m+1:n+m+n_ineq]
    fs = @view primal(ipm.f)[n+m+1:n+m+n_ineq]
    jacls = @view ipm.jacl[n+m+1:n+m+n_ineq]
    zls = @view primal(ipm.zl)[n+m+1:n+m+n_ineq]
    zus = @view primal(ipm.zu)[n+m+1:n+m+n_ineq]
    # Regularization
    r = @view primal(ipm.x)[n+1:n+m]
    pr = @view primal(ipm.p)[n+1:n+m]

    py = dual(ipm.p)
    pzl = dual_lb(ipm.p)
    pzu = dual_ub(ipm.p)

    px .= .-fx .+ zlx .- zux .- jaclx
    # Extrapolation r-block RHS = −∂P/∂r (the crossover keeps only the penalty-gradient term).
    #   conv A:  −∂P/∂r =  φ'(−ρr)   = dphi(kernel, −ρr)   (cosh ⇒ sinh(−ρr) = −sinh(ρr))
    #   conv B:  −∂P/∂r = −ρ φ'(r)   = −ρ·dphi(kernel, r)
    # QuadraticKernel ⇒ −ρr, bit-identical to the original `.-ρ .* r`.
    if solver.ncl.scaled_arg
        pr .= dphi.(Ref(solver.ncl.kernel), .-(ρ .* r))
    else
        pr .= .-ρ .* dphi.(Ref(solver.ncl.kernel), r)
    end
    ps .= .-fs .+ zls .- zus .- jacls
    py .= .-ipm.c
    pzl .= (ipm.xl_r .- ipm.x_lr) .* ipm.zl_r .+ μ
    pzu .= (ipm.xu_r .- ipm.x_ur) .* ipm.zu_r .- μ
    return
end


# N.B.: The extrapolation step is described in detail in:
# [1, Section 3.2] Armand, Paul, and Riadh Omheni.
#    "A mixed logarithmic barrier-augmented Lagrangian method for nonlinear optimization."
#    Journal of Optimization Theory and Applications 173.2 (2017): 523-547.
# [2, Algorithm 1] Armand, Paul, Joël Benoist, and Dominique Orban.
#    "From global to local convergence of interior methods for nonlinear optimization."
#    Optimization Methods and Software 28.5 (2013): 1051-1080.
function extrapolation!(solver::NCLSolver{T}) where T
    ipm = solver.ipm
    ρ = solver.ncl.ρk[]
    rho = solver.options.extrapolation_rho
    μk = ipm.mu
    μ_fac, μ_tau, μ_min = solver.options.mu_fac, solver.options.mu_tau, solver.options.mu_min

    # Evaluate model with new rho and mu.
    ipm.obj_val = eval_f_wrapper(ipm, ipm.x)
    eval_grad_f_wrapper!(ipm, ipm.f, ipm.x)
    eval_cons_wrapper!(ipm, ipm.c, ipm.x)
    eval_jac_wrapper!(ipm, ipm.kkt, ipm.x)
    jtprod!(ipm.jacl, ipm.kkt, ipm.y)
    eval_lag_hess_wrapper!(ipm, ipm.kkt, ipm.x, ipm.y)

    # Previous KKT residual is the norm of the RHS
    set_aug_rhs_extrapolation!(solver, ipm.kkt, μk)
    res_p = norm(ipm.p.values, Inf)

    # Update barrier for extrapolation step
    μp = max(min(μk^μ_tau, μ_fac * μk), μ_min)
    τ = max(T(0.995), one(T) - μp)

    # Solve KKT system
    set_aug_diagonal!(ipm.kkt, ipm)
    set_aug_rhs_extrapolation!(solver, ipm.kkt, μp)
    is_solved = inertia_correction!(ipm.inertia_corrector, ipm)

    # If the linear solver has failed, we leave the extrapolation step
    if !is_solved
        return (false, res_p)
    end

    # Fraction to boundary
    alpha_p = get_alpha_max(
        primal(ipm.x),
        primal(ipm.xl),
        primal(ipm.xu),
        primal(ipm.d),
        τ,
    )
    alpha_d = get_alpha_z(
        ipm.zl_r,
        ipm.zu_r,
        dual_lb(ipm.d),
        dual_ub(ipm.d),
        τ,
    )

    # Take full Newton step
    axpy!(alpha_p, primal(ipm.d), primal(ipm.x))
    axpy!(alpha_p, dual(ipm.d), ipm.y)
    ipm.zl_r .+= alpha_d .* dual_lb(ipm.d)
    ipm.zu_r .+= alpha_d .* dual_ub(ipm.d)

    # Update barrier term
    ipm.mu = μk + alpha_p * (μp - μk)

    # Update callbacks
    ipm.obj_val = eval_f_wrapper(ipm, ipm.x)
    eval_grad_f_wrapper!(ipm, ipm.f, ipm.x)
    eval_cons_wrapper!(ipm, ipm.c, ipm.x)
    eval_jac_wrapper!(ipm, ipm.kkt, ipm.x)
    jtprod!(ipm.jacl, ipm.kkt, ipm.y)

    # Compute KKT residual to check if crossover has succeeded
    set_aug_rhs_extrapolation!(solver, ipm.kkt, ipm.mu)
    res_k = norm(ipm.p.values, Inf)

    if res_k <= rho * res_p + T(10) * alpha_p^T(0.2) * μk
        return (true, res_k)
    else
        # If decrease is not sufficient, the solver does not take
        # the full step and we return to the previous iterate.
        ipm.mu = μk
        axpy!(-alpha_p, primal(ipm.d), primal(ipm.x))
        axpy!(-alpha_p, dual(ipm.d), ipm.y)
        ipm.zl_r .-= alpha_d .* dual_lb(ipm.d)
        ipm.zu_r .-= alpha_d .* dual_ub(ipm.d)
        # Refresh values in callback
        ipm.obj_val = eval_f_wrapper(ipm, ipm.x)
        eval_grad_f_wrapper!(ipm, ipm.f, ipm.x)
        eval_cons_wrapper!(ipm, ipm.c, ipm.x)
        return (false, res_p)
    end
end

function solve!(solver::NCLSolver{T}; diag=nothing) where T
    n, m = solver.n, solver.m
    ncl = solver.ncl
    ipm = solver.ipm
    options = solver.options

    options.verbose && _introduce(n, m)

    # Parameters
    ### Penalty ρ
    ncl.ρk[] = options.rho_init
    ρ_max = options.rho_max
    tau_ρ = options.tau_rho
    ### Barrier parameters
    μ = options.mu_init
    μ_min = options.mu_min
    μ_fac = options.mu_fac
    τ = options.mu_tau
    ### Forcing parameters
    γ = T(0.05)
    eps_c = T(10)*μ
    eps_d = μ <= μ_min ? min(options.eps_d_mult * T(100)*μ^(one(T)+γ), options.eps_d_cap) : options.eps_d_mult * T(100)*μ^(one(T)+γ)
    η = T(0.1)   # initial primal feasibility tolerance

    start_time = time()

    ncl_status = INITIAL

    # Unless the parameter `dual_initialized` is set to `true`,
    # MadNLP computes the initial multipliers y0 in `initialize!`.
    initialize!(ipm)
    # Update initial multiplier using multiplier computed by MadNLP in `initialize!`.
    ncl.yk .= ipm.y

    ipm.status = REGULAR
    cnt_it_inner = 0

    x = view(primal(ipm.x), 1:n)
    r = view(primal(ipm.x), 1+n:n+m)

    pr_feas = ipm.inf_pr
    du_feas = get_inf_du(solver)

    is_extrapolated = false
    options.verbose && _log_header()
    flag = " "
    options.verbose && _log_iter(0, flag, 0, ipm.obj_val, pr_feas, du_feas, η, μ, ncl.ρk[])

    iter = 1
    while iter <= options.max_auglag_iter
        # Update parameters in MadNLP
        setup!(
            ipm;
            tol=eps_d,
            μ=μ,
        )

        # Diagnostic: split factorizations per outer iter into extrapolation vs inner-solve.
        nf0 = ipm.cnt.factorization_cnt; k0 = ipm.cnt.k
        if iter >= 2 && options.extrapolation
            is_extrapolated, delta = extrapolation!(solver)
            flag = is_extrapolated ? "+" : " "
        end
        nf_ex = ipm.cnt.factorization_cnt
        maxiter_hit = false
        has_converged = false
        if !is_extrapolated
            status = regular!(ipm)
            maxiter_hit = status == MAXIMUM_ITERATIONS_EXCEEDED
            has_converged = status == SOLVE_SUCCEEDED
            flag = has_converged ? " " : "r"
        end
        ex_fact = nf_ex - nf0
        in_fact = ipm.cnt.factorization_cnt - nf_ex
        in_iters = ipm.cnt.k - k0
        if maxiter_hit
            diag !== nothing && push!(diag, (iter=iter, rho=ncl.ρk[], mu=μ, pr_feas=NaN, eta=η,
                thr=max(options.eta_mult * η, options.feas_tol), ratio=NaN, has_converged=false,
                extrapolated=is_extrapolated, branch=:maxiter,
                extrap_fact=ex_fact, inner_fact=in_fact, inner_iters=in_iters))
            ncl_status = MAXIMUM_ITERATIONS_EXCEEDED
            break
        end

        # Get KKT residuals for scaled problem
        pr_feas = norm(r, Inf)
        mu_used = μ; rho_used = ncl.ρk[]; eta_used = η  # capture for diag (branch below mutates μ/ρ)
        du_feas = get_inf_du(
            full(ipm.f), full(ipm.zl), full(ipm.zu), ipm.jacl, 1.0,
        )
        cc_feas = get_inf_compl(
            ipm.x_lr, ipm.xl_r, ipm.zl_r, ipm.xu_r, ipm.x_ur, ipm.zu_r, ipm.mu, 1.0,
        )

        # Update parameters
        if is_extrapolated
            ncl.yk .= ipm.y
            ρ_prev_ex = ncl.ρk[]
            ncl.ρk[] = min(max(one(T) / delta, T(1.1) * ncl.ρk[]), ρ_max)
            μ = ipm.mu  # mu has been updated previously when computing extrapolation step
            branch = :extrap
            options.predict_extrap && (r .*= ρ_prev_ex / ncl.ρk[])  # idea (a): keep ρr bounded across extrapolation
        elseif pr_feas <= max(options.eta_mult * η, options.feas_tol) && has_converged
            ncl.yk .= ipm.y
            μ = max(min(μ^τ, μ_fac * μ), μ_min)
            branch = :mu_reduce
        else
            branch = :rho_bump
            ρ_prev = ncl.ρk[]
            ncl.ρk[] = min(ncl.ρk[] * tau_ρ, ρ_max)
            # Continuation predictor (opt-in): after a ρ-bump, reset the residual to keep the penalty
            # argument ρr ≈ O(1), avoiding the cosh dual-infeasibility spike + linear-descent grind.
            # Smarter-predictor refinements: 3(a) skip the reset once feasibility is close (pr_feas ≤
            # predict_skip_below) so the inner solve nails the final feasibility instead of the predictor
            # over-shrinking r (premature/false SUCC); 3(b) clamp |rᵢ| ≥ predict_floor after the reset so it
            # never shrinks r below the feasibility scale.
            if options.predict_resid
                # 3(a)/3(a)': skip the reset only in the genuinely quad-like regime. 3(a) gates on feasibility
                # (pr_feas) alone; 3(a)' ALSO requires the multiplier mismatch ‖y−yₖ‖∞ to be small. The dual
                # clause protects the large-Δλ (degenerate) niche: there pr_feas can be small while Δλ is huge,
                # and skipping would let ρ·r = τ·asinh(Δλ) spike → blowup. (predict_skip_dual=Inf ⇒ plain 3(a).)
                skip_reset = pr_feas <= options.predict_skip_below &&
                             norm(ipm.y .- ncl.yk, Inf) <= options.predict_skip_dual
                if !skip_reset
                    if options.predict_mode == :multiplier && ncl.scaled_arg
                        # Predicted stationary point (conv A): φ'(−ρr*)=y−yₖ ⟹ r* = −(φ')⁻¹(y−yₖ)/ρ_new.
                        r .= .-inv_dphi.(Ref(ncl.kernel), ipm.y .- ncl.yk) ./ ncl.ρk[]
                    else
                        # Geometric: in branch (c) the subproblem converged, so r ≈ r*(ρ_prev), ρ_prev·r ≈ O(1).
                        r .*= ρ_prev / ncl.ρk[]
                    end
                    options.predict_floor > 0 && (@. r = sign(r) * max(abs(r), options.predict_floor))
                end
            end
        end

        diag !== nothing && push!(diag, (iter=iter, rho=rho_used, mu=mu_used, pr_feas=pr_feas, eta=eta_used,
            thr=max(options.eta_mult * eta_used, options.feas_tol), ratio=pr_feas/max(options.eta_mult * eta_used, options.feas_tol),
            has_converged=has_converged, extrapolated=is_extrapolated, branch=branch,
            extrap_fact=ex_fact, inner_fact=in_fact, inner_iters=in_iters))

        η = min(μ^T(1.1), T(0.1) * μ)
        eps_d = μ <= μ_min ? min(options.eps_d_mult * T(100) * μ^(one(T)+γ), options.eps_d_cap) : options.eps_d_mult * T(100) * μ^(one(T)+γ)
        eps_c = T(10) * μ

        # Log evolution
        ipm_iter = ipm.cnt.k
        obj_val = ipm.obj_val + dot(ncl.yk, r) - _penalty_obj(ncl.kernel, ncl.scaled_arg, ncl.ρk[], r)
        options.verbose && _log_iter(iter, flag, ipm_iter, obj_val, pr_feas, du_feas, η, μ, ncl.ρk[])

        # Check convergence
        if (
            max(du_feas, cc_feas) <= options.opt_tol &&
            pr_feas <= options.feas_tol &&
            get_constr_viol_tol(solver.ncl, r) <= options.constr_viol_tol &&
            (!options.tighten_success || _unscaled_eqfeas(solver.ncl, x) <= options.feas_tol)
        )
            ncl_status = SOLVE_SUCCEEDED
            break
        # Check infeasibility
        elseif (ncl.ρk[] >= options.rho_max) && (pr_feas > options.opt_tol)
            ncl_status = INFEASIBLE_PROBLEM_DETECTED
            break
        end
        iter += 1
    end

    if (iter >= options.max_auglag_iter) || (ipm.status == MAXIMUM_ITERATIONS_EXCEEDED)
        ncl_status = MAXIMUM_ITERATIONS_EXCEEDED
    end

    ipm.cnt.total_time = time() - start_time

    return NCLStats(solver, ncl_status)
end

function madncl(
    nlp::NLPModels.AbstractNLPModel;
    options...
)
    solver = NCLSolver(nlp; options...)
    return solve!(solver)
end

