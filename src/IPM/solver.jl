"""
    madnlp(model::AbstractNLPModel; options...)

Build a [`MadNLPSolver`](@ref) and solve it using
the interior-point method. Return the solution
as a [`MadNLPExecutionStats`](@ref).

"""
function madnlp(model::AbstractNLPModel; kwargs...)
    solver = MadNLPSolver(model; kwargs...)
    return solve!(solver)
end

function initialize!(solver::AbstractMadNLPSolver{T}) where T

    nlp = get_nlp(solver)
    opt = get_opt(solver)

    # Initializing variables
    @trace(get_logger(solver),"Initializing variables.")
    initialize!(
        get_cb(solver),
        get_x(solver),
        get_xl(solver),
        get_xu(solver),
        get_y(solver),
        get_rhs(solver),
        get_ind_ineq(solver);
        tol=opt.bound_relax_factor,
        bound_push=opt.bound_push,
        bound_fac=opt.bound_fac,
    )
    fill!(get_jacl(solver), zero(T))
    fill!(get_zl_r(solver), one(T))
    fill!(get_zu_r(solver), one(T))

    # Initializing scaling factors
    if opt.nlp_scaling
        set_scaling!(
            get_cb(solver),
            get_x(solver),
            get_xl(solver),
            get_xu(solver),
            get_y(solver),
            get_rhs(solver),
            get_ind_ineq(solver),
            opt.nlp_scaling_max_gradient
        )
    end

    # Initializing KKT system
    initialize!(get_kkt(solver))

    # Initializing jacobian and gradient
    eval_jac_wrapper!(solver, get_kkt(solver), get_x(solver))
    eval_grad_f_wrapper!(solver, get_f(solver),get_x(solver))


    @trace(get_logger(solver),"Initializing constraint duals.")
    if !get_opt(solver).dual_initialized
        initialize_dual(solver, opt.dual_initialization_method)
    end

    # Initializing
    set_obj_val!(solver, eval_f_wrapper(solver, get_x(solver)))
    eval_cons_wrapper!(solver, get_c(solver), get_x(solver))
    eval_lag_hess_wrapper!(solver, get_kkt(solver), get_x(solver), get_y(solver))

    theta = get_theta(get_c(solver))
    set_theta_max!(solver, T(1e4)*max(one(T),theta))
    set_theta_min!(solver, T(1e-4)*max(one(T),theta))
    set_mu!(solver, get_opt(solver).barrier.mu_init)
    set_tau!(solver, max(get_opt(solver).tau_min,one(T)-get_opt(solver).barrier.mu_init))
    push!(get_filter(solver), (get_theta_max(solver),-T(Inf)))

    return REGULAR
end

abstract type DualInitializeOptions end
struct DualInitializeSetZero <: DualInitializeOptions end
struct DualInitializeLeastSquares <: DualInitializeOptions end

function initialize_dual(solver::AbstractMadNLPSolver{T}, ::Type{DualInitializeSetZero}) where T
    fill!(get_y(solver), zero(T))
end
function initialize_dual(solver::AbstractMadNLPSolver{T}, ::Type{DualInitializeLeastSquares}) where T
    set_initial_rhs!(solver, get_kkt(solver))
    factorize_wrapper!(solver)
    is_solved = solve_refine_wrapper!(
        get_d(solver), solver, get_p(solver), get__w4(solver)
    )
    if !is_solved || (norm(dual(get_d(solver)), T(Inf)) > get_opt(solver).constr_mult_init_max)
        fill!(get_y(solver), zero(T))
    else
        copyto!(get_y(solver), dual(get_d(solver)))
    end
end

function reinitialize!(solver::AbstractMadNLPSolver)
    variable(get_x(solver)) .= get_x0(get_nlp(solver))

    set_obj_val!(solver, eval_f_wrapper(solver, get_x(solver)))
    eval_grad_f_wrapper!(solver, get_f(solver), get_x(solver))
    eval_cons_wrapper!(solver, get_c(solver), get_x(solver))
    eval_jac_wrapper!(solver, get_kkt(solver), get_x(solver))
    eval_lag_hess_wrapper!(solver, get_kkt(solver), get_x(solver), get_y(solver))

    theta = get_theta(get_c(solver))
    set_theta_max!(solver, 1e4*max(1,theta))
    set_theta_min!(solver, 1e-4*max(1,theta))
    set_mu!(solver, get_opt(solver).barrier.mu_init)
    set_tau!(solver, max(get_opt(solver).tau_min,1-get_opt(solver).barrier.mu_init))
    empty!(get_filter(solver))
    push!(get_filter(solver), (get_theta_max(solver),-Inf))

    return REGULAR
end

# major loops ---------------------------------------------------------
"""
    solve!(
        solver::AbstractMadNLPSolver,
        stats::MadNLPExecutionStats;
        kwargs...
    )

Solve the problem formulated inside `solver`. Return the solution
in `stats` by modifying inplace the values. The options are specified in `kwargs`.

"""
solve!(solver::AbstractMadNLPSolver; kwargs...) = solve!(solver, MadNLPExecutionStats(solver); kwargs...)
function solve!(
    solver::AbstractMadNLPSolver,
    stats::MadNLPExecutionStats;
    kwargs...
)
    if !isempty(kwargs)
        @warn(get_logger(solver),"The options set during resolve may not have an effect")
        set_options!(get_opt(solver), kwargs)
    end

    # If the problem has no free variable, do nothing
    if get_n(solver) == 0
        update!(stats, solver)
        return stats
    end

    try
        if get_status(solver) == INITIAL
            @notice(get_logger(solver),"This is $(introduce()), running with $(introduce(get_kkt(solver).linear_solver))\n")
            print_init(solver)
            set_status!(solver, initialize!(solver))
        else # resolving the problem
            set_status!(solver, reinitialize!(solver))
        end

        if is_async(get_kkt(solver).linear_solver)
            @trace(get_logger(solver), "linear solver time is not available for asynchronous linear solvers.")
            get_cnt(solver).linear_solver_time = NaN
        end

        while get_status(solver) >= REGULAR
            get_status(solver) == REGULAR && (set_status!(solver, regular!(solver)))
            get_status(solver) == RESTORE && (set_status!(solver, restore!(solver)))
            get_status(solver) == ROBUST && (set_status!(solver, robust!(solver)))
        end
    catch e
        if e isa InvalidNumberException
            if e.callback == :obj
                set_status!(solver, INVALID_NUMBER_OBJECTIVE)
            elseif e.callback == :grad
                set_status!(solver, INVALID_NUMBER_GRADIENT)
            elseif e.callback == :cons
                set_status!(solver, INVALID_NUMBER_CONSTRAINTS)
            elseif e.callback == :jac
                set_status!(solver, INVALID_NUMBER_JACOBIAN)
            elseif e.callback == :hess
                set_status!(solver, INVALID_NUMBER_HESSIAN_LAGRANGIAN)
            else
                set_status!(solver, INVALID_NUMBER_DETECTED)
            end
        elseif e isa NotEnoughDegreesOfFreedomException
            set_status!(solver, NOT_ENOUGH_DEGREES_OF_FREEDOM)
        elseif e isa LinearSolverException
            set_status!(solver, ERROR_IN_STEP_COMPUTATION;)
            get_opt(solver).rethrow_error && rethrow(e)
        elseif e isa InterruptException
            set_status!(solver, USER_REQUESTED_STOP)
            get_opt(solver).rethrow_error && rethrow(e)
        else
            set_status!(solver, INTERNAL_ERROR)
            get_opt(solver).rethrow_error && rethrow(e)
        end
    finally
        get_cnt(solver).total_time = time() - get_cnt(solver).start_time
        if !(get_status(solver) < SOLVE_SUCCEEDED)
            print_summary(solver)
        end
        @notice(get_logger(solver),"$(Base.text_colors[color_status(get_status(solver))])EXIT: $(get_status_output(get_status(solver), get_opt(solver)))$(Base.text_colors[:normal])")
        get_opt(solver).disable_garbage_collector &&
            (GC.enable(true); @warn(get_logger(solver),"Julia garbage collector is turned back on"))
        finalize(get_logger(solver))

        update!(stats,solver)
    end


    return stats
end

color_status(status::Status) =
    status <= SOLVE_SUCCEEDED ? :green :
    status <= SOLVED_TO_ACCEPTABLE_LEVEL ? :blue : :red


# Advance the penalty parameter μ_P via the handler's pluggable schedule, once per iteration.
# μ_P and the barrier μ_B are DECOUPLED: the schedule gets the whole solver and may move μ_P by
# any rule (a function of μ_B, of the constraint residual, of the iteration count, …). Its sole
# job is to set `eh.muP[]`; the framework below handles the consequences of the move.
#
# We treat (μ_P, μ_B) as a joint barrier/penalty state. Whenever μ_P moves, the penalized filter
# merit `varphi` changes, so — exactly as MadNLP resets the filter on a μ_B move (barrier.jl) —
# we reset the filter and re-cache the penalty terms at the new μ_P. No-op for non-penalty
# treatments and for the `FixedPenalty` schedule (μ_P never moves).
update_penalty_mu!(solver::AbstractMadNLPSolver) = _update_penalty_mu!(get_cb(solver).equality_handler, solver)
_update_penalty_mu!(::AbstractEqualityTreatment, solver) = nothing
function _update_penalty_mu!(eh::KernelPenaltyEquality, solver::AbstractMadNLPSolver{T}) where T
    changed = update_penalty!(eh.schedule, eh, solver)::Bool   # did φ_A move? (μ_P OR the AL multiplier λ)
    if changed
        # φ_A moved (μ_P and/or λ) ⇒ the penalized merit varphi changed ⇒ the filter is stale. Reset it
        # exactly like a μ_B update does (barrier.jl): the anchor (theta_max, -Inf) is penalty-independent
        # (theta = ‖c-s-rhs‖₁), so it stays valid for ANY φ_A move.
        empty!(get_filter(solver))
        push!(get_filter(solver), (get_theta_max(solver), -T(Inf)))
        # Re-cache the penalty terms at the NEW φ_A so the whole Newton/line-search step is consistent:
        # (i) slack(f) = φ'_A(s) = φ'(s;μ_P) + λ feeds the Newton RHS (set_aug_rhs!) and the line-search
        # directional derivative; (ii) get_obj_val is the Armijo baseline varphi, re-evaluated so it matches
        # the trial merits. (D_s = φ''_A = μ_P in set_aug_diagonal! reads eh.muP[] live, so it needs no refresh;
        # the λ·s term is linear ⇒ Hessian unchanged.)
        penalty_gradient!(solver, eh, slack(get_f(solver)), slack(get_x(solver)))
        set_obj_val!(solver, eval_f_wrapper(solver, get_x(solver)))
    end
    return
end
update_penalty!(::FixedPenalty, eh, solver) = false   # never moves φ_A

# Static (fixed-parameter) subproblem-gated continuation (see StaticContinuation in
# equality_kernels.jl). Bump μ_P only when the penalty-barrier subproblem is solved
# (get_inf_barrier ≤ ε_P(μ_P)) AND the equality slack is small enough AT THE NEXT μ_P so the cosh
# argument μ_P·s stays bounded right after the bump. get_inf_barrier already folds in φ'(s) via
# slack(f). Decoupled from μ_B: the optimality tolerance depends on μ_P (default ε_P = 1/μ_P).
function update_penalty!(sched::StaticContinuation, eh::KernelPenaltyEquality, solver::AbstractMadNLPSolver{T}) where T
    μ = eh.muP[]
    if μ < sched.muP_max
        μ_next  = min(T(sched.muP_max), max(T(sched.kappa_P) * μ, μ^T(sched.theta_P)))  # deterministic NEXT μ_P
        E_opt   = get_inf_barrier(solver)                                    # subproblem optimality error
        s_inf   = norm(view(slack(get_x(solver)), eh.ind_eqslack), Inf)      # ‖s_E‖∞ (GPU-safe)
        opt_tol = T(sched.opt_tol_coef) * μ^T(sched.opt_tol_exp)             # ε_P(μ_P) at CURRENT μ_P
        # slack gate at the NEXT μ_P ⇒ μ_next·s ≤ s_thresh, bounding the cosh argument AFTER the bump:
        if E_opt <= opt_tol && s_inf <= T(sched.s_thresh) / μ_next
            eh.muP[] = μ_next
            sched.n_bumps[] += 1
            sched.muP_cur[] = eh.muP[]
            return true                     # μ_P bumped ⇒ φ_A moved
        end
    end
    sched.muP_cur[] = eh.muP[]
    return false
end

# Augmented Lagrangian (Hestenes–Powell method of multipliers) on the free equality slack — corrected
# LANCELOT/Conn–Gould–Toint outer iteration that closes the certification gap. An OUTER step fires once
# the inner penalty-barrier subproblem is solved to the current inner tol ω_k (get_inf_barrier ≤ ω_k)
# OR has genuinely PLATEAUED (no get_inf_barrier improvement for stall_patience iters — a true
# no-progress test, NOT a fixed iteration count). At each outer step we take the Hestenes–Powell step
# λ ← clip(λ + μ_P·s) (unless it is a no-op), and SEPARATELY grow μ_P only on a CONSERVATIVE windowed
# feasibility stall. Slack-row stationarity λ + μ_P·s = y ⇒ λ→y drives s=(y−λ)/μ_P → 0 at FINITE μ_P.
# Keeping μ_P moderate keeps D_s = μ_P (and del_w) well-conditioned. Hessian D_s = μ_P unchanged (the
# λ·s term is linear), so the condensed system stays SPD.
#
# Why the OLD logic failed (certification gap):
#  • the freeze `s_inf ≤ tol && return false` halted λ BEFORE it reached y → the slack-row residual
#    ‖λ+μ_P·s−y‖ (part of get_inf_du) floored above tol (mode A); and
#  • the mutually-exclusive `improved-4× ? update-λ : bump-μ_P` mis-fired: once ‖s‖ plateaued at the
#    fixed-λ equilibrium it BUMPED μ_P (→1e5–1e6) instead of updating λ, inflating del_w (mode A) and
#    the condensed-system dynamic range so the inner x-solve crawled and the barrier stalled (mode B).
# The fix keeps μ_P moderate (windowed bump + genuine-plateau gate) so neither floor appears, and
# keeps updating λ→y (no premature freeze) so the slack-row stationarity and the slack both reach tol.
function update_penalty!(sched::AugmentedLagrangian, eh::KernelPenaltyEquality, solver::AbstractMadNLPSolver{T}) where T
    μ     = eh.muP[]
    tol   = T(get_opt(solver).tol)
    sview = view(slack(get_x(solver)), eh.ind_eqslack)
    s_inf = isempty(eh.ind_eqslack) ? zero(T) : norm(sview, Inf)
    sched.muP_cur[] = Float64(μ)

    # Residual split. inf_du = max(x-row stationarity, slack-row stationarity). The schedule (λ-update
    # and μ_P-bump) can ONLY move the SLACK-row stationarity stat = ‖λ+μ_P·s−y‖∞; the x-row/barrier
    # part it cannot. `slack_binding` ⇒ the slack row is a meaningful chunk of inf_du, so acting can
    # help; otherwise acting just churns the filter (and, for a bump, wrecks the D_s=μ_P conditioning).
    idu  = get_inf_du(solver)
    yv   = view(get_y(solver), eh.ind_eqslack)
    stat = isempty(eh.ind_eqslack) ? zero(T) :
           mapreduce((l, si, yi) -> abs(l + μ * si - yi), max, eh.lambda, sview, yv; init = zero(T))
    slack_binding = stat > T(0.01) * idu

    # ── AL fixed point ⇒ freeze φ_A (no λ move, no μ_P bump, no filter reset) ───────────────────────
    # Freeze when equality feasibility is met (‖s‖∞ ≤ tol) AND the schedule can no longer usefully
    # reduce inf_du: (i) ‖μ_P·s‖∞ ≤ tol (λ has converged to y; Hestenes step is a no-op), or
    # (ii) get_inf_du ≤ tol (success imminent), or (iii) !slack_binding (inf_du is x-row/barrier
    # dominated; a λ-update cannot lower it and would only churn the filter). This REPLACES the old
    # `s_inf ≤ tol`-only freeze, which stranded λ short of y (the certification gap). In mode A,
    # slack_binding holds (stat ≈ inf_du) so λ keeps updating until inf_du ≤ tol. The still-infeasible
    # x-row stall (s > tol) is handled at the gate below, never frozen as a converged point.
    s_inf <= tol && (μ * s_inf <= tol || idu <= tol || !slack_binding) && return false

    # ── Inner-subproblem convergence / genuine-stall test ──────────────────────────────────────────
    # Track the best (min) get_inf_barrier of the current inner solve. While it keeps improving (≥10%)
    # the inner solve is making progress — do NOT take an outer step (forcing a step on a noisy
    # mid-descent s spuriously inflates μ_P). Take an outer step only when the inner subproblem is
    # solved to ω_k, OR has genuinely PLATEAUED (no improvement for stall_patience iters) AND the slack
    # row is still the binding blocker. The plateau net breaks the certification deadlock (when μ_P·s
    # and λ−y nearly cancel, inf_du floors ABOVE ω_k and a pure ω_k gate deadlocks): on a slack-bound
    # plateau we FORCE the corrective λ-update at the equilibrium s (the floor). But on an x-row-bound
    # plateau (a stalled x-solve) forcing an outer step only churns the filter and prevents the x-solve
    # from un-stalling — so we leave φ_A frozen and let the IPM/barrier work (the moderate-μ_P bet).
    E_inner = Float64(get_inf_barrier(solver))
    if E_inner < 0.9 * sched.e_best[]
        sched.e_best[]    = E_inner
        sched.stall_cnt[] = 0
    else
        sched.stall_cnt[] += 1
    end
    inner_solved  = E_inner <= sched.omega[]
    inner_stalled = sched.stall_cnt[] >= sched.stall_patience && slack_binding
    (inner_solved || inner_stalled) || return false

    sched.e_best[]    = Inf      # restart inner-progress tracking for the next subproblem
    sched.stall_cnt[] = 0
    changed = false

    # ── (1) Hestenes–Powell multiplier update — DECOUPLED from (2) ──────────────────────────────────
    # Skip only when the step is a no-op (‖μ_P·s‖∞ ≤ tol ⇒ λ already at y to tol): avoids a needless
    # filter reset / endgame thrash. Otherwise update — this drives λ→y, collapses the slack-row
    # stationarity ‖λ+μ_P·s−y‖, and (via s=(y−λ)/μ_P) the equality slack, at FINITE μ_P.
    if μ * s_inf > tol
        @views eh.lambda .= eh.lambda .+ μ .* sview
        isfinite(sched.lambda_max) && clamp!(eh.lambda, -T(sched.lambda_max), T(sched.lambda_max))
        sched.n_updates[] += 1
        changed = true
    end

    # ── (2) Penalty increase — CONSERVATIVE, windowed, slack-row-gated, NEVER instead of (1) ────────
    # Maintain an anchor (s_ref, n_anchor). Each outer step: if ‖s‖ contracted below eta_lambda·s_ref,
    # re-anchor (the multiplier iteration is converging — DO NOT bump). Bump μ_P only when ‖s‖ fails to
    # contract for `bump_patience` consecutive λ-updates (genuinely stalled at this μ_P) AND the slack
    # row is the binding blocker. Growing μ_P when inf_du is x-row-dominated is futile (the slack row
    # λ+μ_P·s≈y is already satisfied — feasibility is gated by a stalled x-solve, not a weak penalty)
    # and wrecks the D_s=μ_P conditioning — the exact pathology that floored inf_du at μ_P=1e5–1e6.
    # Keeping μ_P moderate instead lets the better-conditioned x-solve un-stall (the mode-B mechanism),
    # and kills the early-transient false bumps that previously ballooned μ_P.
    if s_inf > tol
        if s_inf <= T(sched.eta_lambda) * T(sched.s_ref[])           # contracting → re-anchor, no bump
            sched.s_ref[]    = Float64(s_inf)
            sched.n_anchor[] = sched.n_updates[]
        elseif slack_binding && (sched.n_updates[] - sched.n_anchor[]) >= sched.bump_patience && μ < sched.muP_max
            μ_next = min(T(sched.muP_max), max(T(sched.kappa_P) * μ, μ^T(sched.theta_P)))
            if μ_next != μ
                eh.muP[] = μ_next
                sched.n_bumps[] += 1
                changed = true
            end
            sched.s_ref[]    = Float64(s_inf)                        # re-anchor after a bump
            sched.n_anchor[] = sched.n_updates[]
        end
    else
        sched.s_ref[]    = Float64(s_inf)                            # feasible → reset the anchor
        sched.n_anchor[] = sched.n_updates[]
    end

    # ── (3) Tighten the inner tolerance ω_k toward ~tol (decoupled from μ_P) ────────────────────────
    # The old ε_P = 1/μ_P could never tighten below 1/μ_P, capping ‖λ−y‖ at ~1/μ_P and forcing μ_P→∞.
    # Geometric tightening; the stall net guarantees progress regardless of the exact ω_k.
    sched.omega[]   = max(Float64(tol) / 10, sched.omega[] * sched.beta_omega)
    sched.muP_cur[] = Float64(eh.muP[])
    return changed
end

function regular!(solver::AbstractMadNLPSolver{T}) where T
    while true
        if (get_cnt(solver).k!=0 && !get_opt(solver).jacobian_constant)
            eval_jac_wrapper!(solver, get_kkt(solver), get_x(solver)) # refresh Jacobian at current x
        end

        jtprod!(get_jacl(solver), get_kkt(solver), get_y(solver)) # jacl = J^\top y
        sd = get_sd(get_y(solver),get_zl_r(solver),get_zu_r(solver),T(get_opt(solver).s_max)) # ipopt scaling factor
        sc = get_sc(get_zl_r(solver),get_zu_r(solver),T(get_opt(solver).s_max)) # ipopt scaling factor
        set_inf_pr!(solver, get_inf_pr(get_c(solver))) # primal infeasibility
        set_inf_du!(solver, get_inf_du(
            full(get_f(solver)),
            full(get_zl(solver)),
            full(get_zu(solver)),
            get_jacl(solver),
            sd,
        )) # dual infeasibility
        set_inf_compl!(solver, get_inf_compl(solver, sc; mu=zero(T))) # complementarity for overall problem

        print_iter(solver)

        # evaluate termination criteria
        @trace(get_logger(solver),"Evaluating termination criteria.")
        !(get_intermediate_callback(solver)(solver, UserCallbackRegular()) :: Bool) && return USER_REQUESTED_STOP
        # success requires the scaled KKT AND true equality feasibility ‖s_E‖∞≤tol (the freed penalty
        # slack is not in get_inf_total; penalty_eq_feasible is a no-op for non-penalty treatments).
        get_inf_total(solver) <= get_opt(solver).tol &&
            penalty_eq_feasible(get_cb(solver).equality_handler, solver, get_opt(solver).tol) &&
            return SOLVE_SUCCEEDED # success
        (get_inf_total(solver) <= get_opt(solver).acceptable_tol &&
         penalty_eq_feasible(get_cb(solver).equality_handler, solver, get_opt(solver).acceptable_tol)) ?
            (get_cnt(solver).acceptable_cnt < get_opt(solver).acceptable_iter ?
            get_cnt(solver).acceptable_cnt+=1 : return SOLVED_TO_ACCEPTABLE_LEVEL) : (get_cnt(solver).acceptable_cnt = 0) # if an acceptable tolerance enough times in a row is set, check.
        get_inf_total(solver) >= get_opt(solver).diverging_iterates_tol && return DIVERGING_ITERATES # diverged if E exceeds some large constant
        get_cnt(solver).k>=get_opt(solver).max_iter && return MAXIMUM_ITERATIONS_EXCEEDED # max iter
        time()-get_cnt(solver).start_time>=get_opt(solver).max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED # max time 

        # evaluate Hessian
        if (get_cnt(solver).k!=0 && !get_opt(solver).hessian_constant)
            eval_lag_hess_wrapper!(solver, get_kkt(solver), get_x(solver), get_y(solver))
        end

        # update the barrier parameter
        @trace(get_logger(solver),"Updating the barrier parameter.")
        update_barrier!(get_opt(solver).barrier, solver, sc)
        # update the penalty parameter (uses the just-updated μ_B; no-op unless a
        # KernelPenaltyEquality handler with a non-fixed schedule is active)
        update_penalty_mu!(solver)

        # factorize the KKT system and solve Newton step
        @trace(get_logger(solver),"Computing the Newton step.")
        set_aug_diagonal!(get_kkt(solver),solver)
        set_aug_rhs!(solver, get_kkt(solver), get_c(solver), get_mu(solver))
        dual_inf_perturbation!(primal(get_p(solver)),get_ind_llb(solver),get_ind_uub(solver),get_mu(solver),get_opt(solver).kappa_d)
        inertia_correction!(get_inertia_corrector(solver), solver) || return ROBUST

        @trace(get_logger(solver),"Backtracking line search initiated.")
        status = filter_line_search!(solver)
        if status != LINESEARCH_SUCCEEDED
            return status
        end

        @trace(get_logger(solver),"Updating primal-dual variables.")
        copyto!(full(get_x(solver)), full(get_x_trial(solver)))
        copyto!(get_c(solver), get_c_trial(solver))
        set_obj_val!(solver, get_obj_val_trial(solver))
        adjust_boundary!(get_x_lr(solver),get_xl_r(solver),get_x_ur(solver),get_xu_r(solver),get_mu(solver))

        axpy!(get_alpha(solver),dual(get_d(solver)),get_y(solver))

        get_zl_r(solver) .+= get_alpha_z(solver) .* dual_lb(get_d(solver))
        get_zu_r(solver) .+= get_alpha_z(solver) .* dual_ub(get_d(solver))
        reset_bound_dual!(
            primal(get_zl(solver)),
            primal(get_x(solver)),
            primal(get_xl(solver)),
            get_mu(solver),get_opt(solver).kappa_sigma,
        )
        reset_bound_dual!(
            primal(get_zu(solver)),
            primal(get_xu(solver)),
            primal(get_x(solver)),
            get_mu(solver),get_opt(solver).kappa_sigma,
        )

        eval_grad_f_wrapper!(solver, get_f(solver),get_x(solver))

        get_cnt(solver).k+=1
        @trace(get_logger(solver),"Proceeding to the next interior point iteration.")
    end
end

function restore!(solver::AbstractMadNLPSolver{T}) where T
    set_del_w!(solver, zero(T))
    # Backup the previous primal iterate
    copyto!(primal(get__w1(solver)), full(get_x(solver)))
    copyto!(dual(get__w1(solver)), get_y(solver))
    copyto!(dual(get__w2(solver)), get_c(solver))

    F = get_F(
        get_c(solver),
        primal(get_f(solver)),
        primal(get_zl(solver)),
        primal(get_zu(solver)),
        get_jacl(solver),
        get_x_lr(solver),
        get_xl_r(solver),
        get_zl_r(solver),
        get_xu_r(solver),
        get_x_ur(solver),
        get_zu_r(solver),
        get_mu(solver),
    )
    set_alpha_z!(solver, zero(T))
    set_ftype!(solver, "R")
    while true
        alpha_max = get_alpha_max(
            primal(get_x(solver)),
            primal(get_xl(solver)),
            primal(get_xu(solver)),
            primal(get_d(solver)),
            get_tau(solver),
        )
        set_alpha!(solver,min(
            alpha_max,
            get_alpha_z(get_zl_r(solver),get_zu_r(solver),dual_lb(get_d(solver)),dual_ub(get_d(solver)),get_tau(solver)),
        ))

        axpy!(get_alpha(solver), primal(get_d(solver)), full(get_x(solver)))
        axpy!(get_alpha(solver), dual(get_d(solver)), get_y(solver))
        get_zl_r(solver) .+= get_alpha(solver) .* dual_lb(get_d(solver))
        get_zu_r(solver) .+= get_alpha(solver) .* dual_ub(get_d(solver))

        eval_cons_wrapper!(solver,get_c(solver),get_x(solver))
        eval_grad_f_wrapper!(solver,get_f(solver),get_x(solver))
        set_obj_val!(solver, eval_f_wrapper(solver,get_x(solver)))

        !get_opt(solver).jacobian_constant && eval_jac_wrapper!(solver,get_kkt(solver),get_x(solver))
        jtprod!(get_jacl(solver),get_kkt(solver),get_y(solver))

        F_trial = get_F(
            get_c(solver),
            primal(get_f(solver)),
            primal(get_zl(solver)),
            primal(get_zu(solver)),
            get_jacl(solver),
            get_x_lr(solver),
            get_xl_r(solver),
            get_zl_r(solver),
            get_xu_r(solver),
            get_x_ur(solver),
            get_zu_r(solver),
            get_mu(solver),
        )
        if F_trial > get_opt(solver).soft_resto_pderror_reduction_factor*F
            copyto!(primal(get_x(solver)), primal(get__w1(solver)))
            copyto!(get_y(solver), dual(get__w1(solver)))
            copyto!(get_c(solver), dual(get__w2(solver))) # backup the previous primal iterate
            return ROBUST
        end

        adjust_boundary!(get_x_lr(solver),get_xl_r(solver),get_x_ur(solver),get_xu_r(solver),get_mu(solver))

        F = F_trial

        theta = get_theta(get_c(solver))
        varphi= get_varphi(get_obj_val(solver),get_x_lr(solver),get_xl_r(solver),get_xu_r(solver),get_x_ur(solver),get_mu(solver))

        get_cnt(solver).k+=1
        !(get_intermediate_callback(solver)(solver, UserCallbackRestore()) :: Bool) && return USER_REQUESTED_STOP

        is_filter_acceptable(get_filter(solver),theta,varphi) ? (return REGULAR) : (get_cnt(solver).t+=1)
        get_cnt(solver).k>=get_opt(solver).max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-get_cnt(solver).start_time>=get_opt(solver).max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED


        sd = get_sd(get_y(solver),get_zl_r(solver),get_zu_r(solver),get_opt(solver).s_max)
        sc = get_sc(get_zl_r(solver),get_zu_r(solver),get_opt(solver).s_max)
        set_inf_pr!(solver, get_inf_pr(get_c(solver)))
        set_inf_du!(solver, get_inf_du(
            primal(get_f(solver)),
            primal(get_zl(solver)),
            primal(get_zu(solver)),
            get_jacl(solver),
            sd,
        ))

        set_inf_compl!(solver, get_inf_compl(get_x_lr(solver),get_xl_r(solver),get_zl_r(solver),get_xu_r(solver),get_x_ur(solver),get_zu_r(solver),zero(T),sc))
        set_inf_compl_mu!(solver, get_inf_compl(get_x_lr(solver),get_xl_r(solver),get_zl_r(solver),get_xu_r(solver),get_x_ur(solver),get_zu_r(solver),get_mu(solver),sc))
        print_iter(solver)

        !get_opt(solver).hessian_constant && eval_lag_hess_wrapper!(solver,get_kkt(solver),get_x(solver),get_y(solver))
        set_aug_diagonal!(get_kkt(solver),solver)
        set_aug_rhs!(solver, get_kkt(solver), get_c(solver), get_mu(solver))

        dual_inf_perturbation!(primal(get_p(solver)),get_ind_llb(solver),get_ind_uub(solver),get_mu(solver),get_opt(solver).kappa_d)
        factorize_wrapper!(solver)
        solve_refine_wrapper!(
            get_d(solver), solver, get_p(solver), get__w4(solver)
        )

        set_ftype!(solver, "f")
    end
end

function robust!(solver::AbstractMadNLPSolver{T}) where T
    initialize_robust_restorer!(solver)
    RR = get_RR(solver)
    while true
        if !get_opt(solver).jacobian_constant
            eval_jac_wrapper!(solver, get_kkt(solver), get_x(solver))
        end
        jtprod!(get_jacl(solver), get_kkt(solver), get_y(solver))

        # evaluate termination criteria
        @trace(get_logger(solver),"Evaluating restoration phase termination criteria.")
        sd = get_sd(get_y(solver),get_zl_r(solver),get_zu_r(solver),get_opt(solver).s_max)
        sc = get_sc(get_zl_r(solver),get_zu_r(solver),get_opt(solver).s_max)
        set_inf_pr!(solver, get_inf_pr(get_c(solver)))
        set_inf_du!(solver, get_inf_du(
            primal(get_f(solver)),
            primal(get_zl(solver)),
            primal(get_zu(solver)),
            get_jacl(solver),
            sd,
        ))
        set_inf_compl!(solver, get_inf_compl(get_x_lr(solver),get_xl_r(solver),get_zl_r(solver),get_xu_r(solver),get_x_ur(solver),get_zu_r(solver),zero(T),sc))

        # Robust restoration phase error
        RR.inf_pr_R = get_inf_pr_R(get_c(solver),RR.pp,RR.nn)
        RR.inf_du_R = get_inf_du_R(RR.f_R,get_y(solver),primal(get_zl(solver)),primal(get_zu(solver)),get_jacl(solver),RR.zp,RR.zn,get_opt(solver).rho,sd)
        RR.inf_compl_R = get_inf_compl_R(
            get_x_lr(solver),get_xl_r(solver),get_zl_r(solver),get_xu_r(solver),get_x_ur(solver),get_zu_r(solver),RR.pp,RR.zp,RR.nn,RR.zn,zero(T),sc)

        print_iter(solver;is_resto=true)
        !(get_intermediate_callback(solver)(solver, UserCallbackRobust()) :: Bool) && return USER_REQUESTED_STOP

        max(RR.inf_pr_R,RR.inf_du_R,RR.inf_compl_R) <= get_opt(solver).tol && return INFEASIBLE_PROBLEM_DETECTED
        get_cnt(solver).k>=get_opt(solver).max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-get_cnt(solver).start_time>=get_opt(solver).max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED

        # update the barrier parameter
        @trace(get_logger(solver),"Updating restoration phase barrier parameter.")
        _update_monotone_RR!(get_opt(solver).barrier, solver, sc)

        # compute the Newton step
        if !get_opt(solver).hessian_constant
            eval_lag_hess_wrapper!(solver, get_kkt(solver), get_x(solver), get_y(solver); is_resto=true)
        end

        # without inertia correction,
        @trace(get_logger(solver),"Solving restoration phase primal-dual system.")
        set_aug_RR!(get_kkt(solver), solver, RR)
        set_aug_rhs_RR!(solver, get_kkt(solver), RR, get_opt(solver).rho)
        inertia_correction!(get_inertia_corrector(solver), solver) || return RESTORATION_FAILED
        finish_aug_solve_RR!(
            RR.dpp,RR.dnn,RR.dzp,RR.dzn,get_y(solver),dual(get_d(solver)),
            RR.pp,RR.nn,RR.zp,RR.zn,RR.mu_R,get_opt(solver).rho
        )

        # filter start
        @trace(get_logger(solver),"Backtracking line search initiated.")
        status = filter_line_search_RR!(solver)
        if status != LINESEARCH_SUCCEEDED
            return status
        end

        @trace(get_logger(solver),"Updating primal-dual variables.")
        copyto!(full(get_x(solver)), full(get_x_trial(solver)))
        copyto!(get_c(solver), get_c_trial(solver))
        copyto!(RR.pp, RR.pp_trial)
        copyto!(RR.nn, RR.nn_trial)

        RR.obj_val_R=RR.obj_val_R_trial
        set_f_RR!(solver,RR)

        axpy!(get_alpha(solver), dual(get_d(solver)), get_y(solver))
        axpy!(get_alpha_z(solver), RR.dzp,RR.zp)
        axpy!(get_alpha_z(solver), RR.dzn,RR.zn)

        get_zl_r(solver) .+= get_alpha_z(solver) .* dual_lb(get_d(solver))
        get_zu_r(solver) .+= get_alpha_z(solver) .* dual_ub(get_d(solver))

        reset_bound_dual!(
            primal(get_zl(solver)),
            primal(get_x(solver)),
            primal(get_xl(solver)),
            RR.mu_R, get_opt(solver).kappa_sigma,
        )
        reset_bound_dual!(
            primal(get_zu(solver)),
            primal(get_xu(solver)),
            primal(get_x(solver)),
            RR.mu_R, get_opt(solver).kappa_sigma,
        )
        reset_bound_dual!(RR.zp,RR.pp,RR.mu_R,get_opt(solver).kappa_sigma)
        reset_bound_dual!(RR.zn,RR.nn,RR.mu_R,get_opt(solver).kappa_sigma)

        adjust_boundary!(get_x_lr(solver),get_xl_r(solver),get_x_ur(solver),get_xu_r(solver),get_mu(solver))

        # check if going back to regular phase
        @trace(get_logger(solver),"Checking if going back to regular phase.")
        set_obj_val!(solver, eval_f_wrapper(solver, get_x(solver)))
        eval_grad_f_wrapper!(solver, get_f(solver), get_x(solver))
        theta = get_theta(get_c(solver))
        varphi= get_varphi(get_obj_val(solver),get_x_lr(solver),get_xl_r(solver),get_xu_r(solver),get_x_ur(solver),get_mu(solver))

        if is_filter_acceptable(get_filter(solver),theta,varphi) &&
            theta <= get_opt(solver).required_infeasibility_reduction * RR.theta_ref

            @trace(get_logger(solver),"Going back to the regular phase.")
            set_initial_rhs!(solver, get_kkt(solver))
            initialize!(get_kkt(solver))

            factorize_wrapper!(solver)
            solve_refine_wrapper!(
                get_d(solver), solver, get_p(solver), get__w4(solver)
            )
            if norm(dual(get_d(solver)), Inf)>get_opt(solver).constr_mult_init_max
                fill!(get_y(solver), zero(T))
            else
                copyto!(get_y(solver), dual(get_d(solver)))
            end

            get_cnt(solver).k+=1
            get_cnt(solver).t+=1

            return REGULAR
        end

        get_cnt(solver).k>=get_opt(solver).max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-get_cnt(solver).start_time>=get_opt(solver).max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED

        @trace(get_logger(solver),"Proceeding to the next restoration phase iteration.")
        get_cnt(solver).k+=1
        get_cnt(solver).t+=1
    end
end

function second_order_correction(solver::AbstractMadNLPSolver,alpha_max,theta,varphi,
                                 theta_trial,varphi_d,switching_condition::Bool)
    @trace(get_logger(solver),"Second-order correction started.")

    wx = primal(get__w1(solver))
    wy = dual(get__w1(solver))
    copyto!(wy, get_c_trial(solver))
    axpy!(alpha_max, get_c(solver), wy)

    theta_soc_old = theta_trial
    for p=1:get_opt(solver).max_soc
        # compute second order correction
        set_aug_rhs!(solver, get_kkt(solver), wy, get_mu(solver))
        dual_inf_perturbation!(
            primal(get_p(solver)),
            get_ind_llb(solver),get_ind_uub(solver),get_mu(solver),get_opt(solver).kappa_d,
        )
        solve_refine_wrapper!(
            get__w1(solver), solver, get_p(solver), get__w4(solver)
        )
        alpha_soc = get_alpha_max(
            primal(get_x(solver)),
            primal(get_xl(solver)),
            primal(get_xu(solver)),
            wx,get_tau(solver)
        )

        copyto!(primal(get_x_trial(solver)), primal(get_x(solver)))
        axpy!(alpha_soc, wx, primal(get_x_trial(solver)))
        eval_cons_wrapper!(solver, get_c_trial(solver), get_x_trial(solver))
        set_obj_val_trial!(solver, eval_f_wrapper(solver, get_x_trial(solver)))

        theta_soc = get_theta(get_c_trial(solver))
        varphi_soc= get_varphi(get_obj_val_trial(solver),get_x_trial_lr(solver),get_xl_r(solver),get_xu_r(solver),get_x_trial_ur(solver),get_mu(solver))

        !is_filter_acceptable(get_filter(solver),theta_soc,varphi_soc) && break

        if theta <=get_theta_min(solver) && switching_condition
            # Case I
            if is_armijo(varphi_soc,varphi,get_opt(solver).eta_phi,get_alpha(solver),varphi_d)
                @trace(get_logger(solver),"Step in second order correction accepted by armijo condition.")
                set_ftype!(solver, "F")
                set_alpha!(solver, alpha_soc)
                return true
            end
        else
            # Case II
            if is_sufficient_progress(theta_soc,theta,get_opt(solver).gamma_theta,varphi_soc,varphi,get_opt(solver).gamma_phi,has_constraints(solver))
                @trace(get_logger(solver),"Step in second order correction accepted by sufficient progress.")
                set_ftype!(solver, "H")
                set_alpha!(solver, alpha_soc)
                return true
            end
        end

        theta_soc>get_opt(solver).kappa_soc*theta_soc_old && break
        theta_soc_old = theta_soc
    end
    @trace(get_logger(solver),"Second-order correction terminated.")

    return false
end


function inertia_correction!(
    inertia_corrector::InertiaBased,
    solver::AbstractMadNLPSolver{T}
) where {T}

    n_trial = 0
    del_w_prev = zero(T)
    del_c_prev = zero(T)
    set_del_w!(solver, zero(T))
    set_del_c!(solver, zero(T))

    @trace(get_logger(solver),"Inertia-based regularization started.")

    factorize_wrapper!(solver)

    num_pos,num_zero,num_neg = inertia(get_kkt(solver).linear_solver)


    solve_status = if is_inertia_correct(get_kkt(solver), num_pos, num_zero, num_neg)
        # Try a backsolve. If the factorization has failed, solve_refine_wrapper returns false.
        solve_refine_wrapper!(get_d(solver), solver, get_p(solver), get__w4(solver))
    else
        false
    end

    while !solve_status
        @debug(get_logger(solver),"Primal-dual perturbed.")

        if n_trial == 0
            set_del_w!(solver, get_del_w_last(solver)==zero(T) ? get_opt(solver).first_hessian_perturbation :
                max(get_opt(solver).min_hessian_perturbation,get_opt(solver).perturb_dec_fact*get_del_w_last(solver))
                    )
        else
            set_del_w!(solver, get_del_w(solver) * (get_del_w_last(solver)==zero(T) ? get_opt(solver).perturb_inc_fact_first : get_opt(solver).perturb_inc_fact))
            if get_del_w(solver)>get_opt(solver).max_hessian_perturbation
                get_cnt(solver).k+=1
                @debug(get_logger(solver),"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        set_del_c!(solver, should_regularize_dual(get_kkt(solver), num_pos, num_zero, num_neg) ? get_opt(solver).jacobian_regularization_value * get_mu(solver)^(get_opt(solver).jacobian_regularization_exponent) : zero(T))
        regularize_diagonal!(get_kkt(solver), get_del_w(solver) - del_w_prev, get_del_c(solver) - del_c_prev)
        del_w_prev = get_del_w(solver)
        del_c_prev = get_del_c(solver)

        factorize_wrapper!(solver)
        num_pos,num_zero,num_neg = inertia(get_kkt(solver).linear_solver)

        solve_status = if is_inertia_correct(get_kkt(solver), num_pos, num_zero, num_neg)
            solve_refine_wrapper!(get_d(solver), solver, get_p(solver), get__w4(solver))
        else
            false
        end

        n_trial += 1
    end

    get_del_w(solver) != 0 && (set_del_w_last!(solver, get_del_w(solver)))
    return true
end

function inertia_correction!(
    inertia_corrector::InertiaFree,
    solver::AbstractMadNLPSolver{T}
    ) where T
    n_trial = 0
    del_w_prev = zero(T)
    del_c_prev = zero(T)
    set_del_w!(solver, zero(T))
    set_del_c!(solver, zero(T))

    @trace(get_logger(solver),"Inertia-free regularization started.")
    dx = primal(get_d(solver))
    p0 = inertia_corrector.p0
    d0 = inertia_corrector.d0
    t = inertia_corrector.t
    n = primal(d0)
    wx= inertia_corrector.wx
    g = inertia_corrector.g

    set_g_ifr!(solver,g)
    set_aug_rhs_ifr!(solver, get_kkt(solver), p0)

    factorize_wrapper!(solver)

    solve_status = solve_refine_wrapper!(
        d0, solver, p0, get__w3(solver),
    ) && solve_refine_wrapper!(
        get_d(solver), solver, get_p(solver), get__w4(solver),
    )
    copyto!(t,dx)
    axpy!(-1.,n,t)

    while !curv_test(t,n,g,get_kkt(solver),wx,get_opt(solver).inertia_free_tol) || !solve_status
        @debug(get_logger(solver),"Primal-dual perturbed.")
        if n_trial == 0
            set_del_w!(solver, get_del_w_last(solver)==.0 ? get_opt(solver).first_hessian_perturbation :
                max(get_opt(solver).min_hessian_perturbation,get_opt(solver).perturb_dec_fact*get_del_w_last(solver))
                    )
        else
            set_del_w!(solver, get_del_w(solver) * (get_del_w_last(solver)==.0 ? get_opt(solver).perturb_inc_fact_first : get_opt(solver).perturb_inc_fact))
            if get_del_w(solver)>get_opt(solver).max_hessian_perturbation
                get_cnt(solver).k+=1
                @debug(get_logger(solver),"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        set_del_c!(solver, get_opt(solver).jacobian_regularization_value * get_mu(solver)^(get_opt(solver).jacobian_regularization_exponent))
        regularize_diagonal!(get_kkt(solver), get_del_w(solver) - del_w_prev, get_del_c(solver) - del_c_prev)
        del_w_prev = get_del_w(solver)
        del_c_prev = get_del_c(solver)

        factorize_wrapper!(solver)
        solve_status = solve_refine_wrapper!(
            d0, solver, p0, get__w3(solver)
        ) && solve_refine_wrapper!(
            get_d(solver), solver, get_p(solver), get__w4(solver)
        )
        copyto!(t,dx)
        axpy!(-1.,n,t)

        n_trial += 1
    end

    get_del_w(solver) != 0 && (set_del_w_last!(solver, get_del_w(solver)))
    return true
end

function inertia_correction!(
    inertia_corrector::InertiaIgnore,
    solver::AbstractMadNLPSolver{T}
    ) where T

    n_trial = 0
    del_w_prev = zero(T)
    del_c_prev = zero(T)
    set_del_w!(solver, zero(T))
    set_del_c!(solver, zero(T))

    @trace(get_logger(solver),"Inertia-based regularization started.")

    factorize_wrapper!(solver)

    solve_status = solve_refine_wrapper!(
        get_d(solver), solver, get_p(solver), get__w4(solver),
    )
    while !solve_status
        @debug(get_logger(solver),"Primal-dual perturbed.")
        if n_trial == 0
            set_del_w!(solver, get_del_w_last(solver)==zero(T) ? get_opt(solver).first_hessian_perturbation :
                max(get_opt(solver).min_hessian_perturbation,get_opt(solver).perturb_dec_fact*get_del_w_last(solver)))
        else
            set_del_w!(solver, get_del_w(solver) * (get_del_w_last(solver)==zero(T) ? get_opt(solver).perturb_inc_fact_first : get_opt(solver).perturb_inc_fact))
            if get_del_w(solver)>get_opt(solver).max_hessian_perturbation
                get_cnt(solver).k+=1
                @debug(get_logger(solver),"Primal regularization is too big. Switching to restoration phase.")
                return false
            end
        end
        set_del_c!(solver, get_opt(solver).jacobian_regularization_value * get_mu(solver)^(get_opt(solver).jacobian_regularization_exponent))
        regularize_diagonal!(get_kkt(solver), get_del_w(solver) - del_w_prev, get_del_c(solver) - del_c_prev)
        del_w_prev = get_del_w(solver)
        del_c_prev = get_del_c(solver)

        factorize_wrapper!(solver)
        solve_status = solve_refine_wrapper!(
            get_d(solver), solver, get_p(solver), get__w4(solver)
        )
        n_trial += 1
    end
    get_del_w(solver) != 0 && (set_del_w_last!(solver, get_del_w(solver)))
    return true
end

function curv_test(t,n,g,kkt,wx,inertia_free_tol)
    mul_hess_blk!(wx, kkt, t)
    dot(wx,t) + max(dot(wx,n)-dot(g,n),0) - inertia_free_tol*dot(t,t) >=0
end
