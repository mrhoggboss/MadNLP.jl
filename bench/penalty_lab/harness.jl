# μ_P continuation lab — STABLE EXECUTION + DATA-CAPTURE infrastructure.
# (Responsibility: run a config and record the per-iteration solver trajectory. This file
#  rarely changes. μ_P strategies live in schedules.jl; experiment configs in the driver;
#  reporting in report.jl.)
#
# Env-agnostic: works for augmented/MA27, condensed/CHOLMOD|MUMPS, and condensed/cuDSS-GPU.
# The *config* supplies the problem builder, kkt_system, linear_solver, and treatment, so the
# harness never imports a specific linear solver or problem source.
#
# Trajectory capture uses MadNLP's intermediate_callback, which fires at the top of every IPM
# iteration (src/IPM/solver.jl:284 regular / :425 restore / :491 robust), i.e. it snapshots the
# state of iterate k *before* that iteration's barrier/penalty update.

using MadNLP, NLPModels, Printf

# ---- per-iteration recorder (an intermediate_callback) -------------------------------
# Fires at the TOP of every IPM iteration. "State" columns (muB, muP, inf_*, obj, seq_inf)
# describe the current iterate k; "step-outcome" columns (alpha, alpha_z, ftype, del_w, del_c,
# facts_iter, ls_l) describe the step that PRODUCED iterate k (iteration k-1) — the same
# one-iteration-lag convention as MadNLP's own print_iter. facts_iter is the per-iteration delta
# of the cumulative factorization counter (= inertia-correction passes). SOC inner-iteration
# count is not tracked by MadNLP (no counter); ftype = F/H marks an SOC-accepted step.
mutable struct Recorder <: MadNLP.AbstractUserCallback
    rows::Vector{NamedTuple}
    prev_nfact::Int
end
Recorder() = Recorder(NamedTuple[], -1)

# strip the "UserCallback" prefix off the mode type → :Regular / :Restore / :Robust
_phase(mode) = Symbol(replace(string(nameof(typeof(mode))), "UserCallback" => ""))

function (rec::Recorder)(solver, mode)
    cnt   = MadNLP.get_cnt(solver)
    eh    = MadNLP.get_cb(solver).equality_handler
    ispen = eh isa KernelPenaltyEquality
    muP   = ispen ? Float64(eh.muP[]) : NaN
    seqi  = (ispen && !isempty(eh.ind_eqslack)) ?
            Float64(maximum(abs, @view MadNLP.slack(MadNLP.get_x(solver))[eh.ind_eqslack])) : NaN
    nf         = cnt.factorization_cnt
    facts_iter = rec.prev_nfact < 0 ? 0 : nf - rec.prev_nfact   # KKT factorizations of the last iter
    rec.prev_nfact = nf
    push!(rec.rows, (
        k         = cnt.k,
        phase     = _phase(mode),                            # Regular / Restore / Robust
        muB       = Float64(MadNLP.get_mu(solver)),          # barrier parameter
        muP       = muP,                                     # penalty parameter (NaN if no penalty)
        inf_pr    = Float64(MadNLP.get_inf_pr(solver)),      # primal infeas (coupling residual)
        inf_du    = Float64(MadNLP.get_inf_du(solver)),      # dual infeas (stationarity, incl φ'-y)
        inf_compl = Float64(MadNLP.get_inf_compl(solver)),   # complementarity
        obj_int   = Float64(MadNLP.get_obj_val(solver)),     # internal merit objective (incl penalty)
        obj_true  = Float64(MadNLP.true_obj_val(solver)),    # model objective f(x) (penalty stripped)
        seq_inf   = seqi,                                    # ‖s_eq‖∞ (penalty slack magnitude)
        alpha     = Float64(MadNLP.get_alpha(solver)),       # ACCEPTED primal step from the filter line search
        alpha_z   = Float64(MadNLP.get_alpha_z(solver)),     # accepted dual (bound-multiplier) step
        ftype     = string(MadNLP.get_ftype(solver)),        # filter step type: f/h normal, F/H SOC-accepted
        del_w     = Float64(MadNLP.get_del_w(solver)),       # primal Hessian regularization δ_w
        del_c     = Float64(MadNLP.get_del_c(solver)),       # dual (Jacobian) regularization δ_C (>0 ⇒ added)
        n_factor  = nf,                                      # cumulative KKT factorizations
        facts_iter= facts_iter,                             # factorizations this iter (= IC passes; retries = facts_iter-1)
        ls_l      = cnt.l,                                 # filter line-search trial count of the last iter
        restoration_t = cnt.t,                            # cumulative restoration-phase counter
    ))
    return true
end

# ---- equality feasibility ‖c_E(x) - b‖∞ at the returned point (host or device x) -----
function eqfeas(nlp, x)
    m = get_ncon(nlp); m == 0 && return 0.0
    c = similar(x, m); cons!(nlp, x, c)
    lc = get_lcon(nlp); uc = get_ucon(nlp)
    return Float64(maximum(abs.(c .- lc) .* (lc .== uc)))
end

# ---- a single run ---------------------------------------------------------------------
# A config is a NamedTuple: (name, make_nlp, kkt, ls, treatment, [callback, nlp_scaling, tol,
# max_iter, barrier, cleanup]). `make_nlp()` returns a fresh NLPModel; `treatment` is a
# constructed KernelPenaltyEquality(...) instance (or a treatment type/instance); `cleanup(nlp)`
# (default no-op) frees resources, e.g. `finalize` for CUTEst models.
struct RunResult
    name::String
    status::Any
    iters::Int
    n_factor::Int
    time::Float64
    eqfeas::Float64
    obj::Float64          # reported model objective f(x*) (penalty stripped)
    rec::Recorder
end

function run_config(cfg)
    nlp = cfg.make_nlp()
    rec = Recorder()
    cleanup = get(cfg, :cleanup, _ -> nothing)
    try
        local r
        t = @elapsed r = madnlp(nlp;
            callback              = get(cfg, :callback, MadNLP.SparseCallback),
            kkt_system            = cfg.kkt,
            linear_solver         = cfg.ls,
            equality_treatment    = cfg.treatment,
            intermediate_callback = rec,
            nlp_scaling           = get(cfg, :nlp_scaling, false),
            tol                   = get(cfg, :tol, 1e-8),
            max_iter              = get(cfg, :max_iter, 3000),
            print_level           = MadNLP.ERROR,
            (haskey(cfg, :barrier) ? (; barrier = cfg.barrier) : (;))...,
        )
        x = r.solution
        return RunResult(cfg.name, r.status, r.iter, r.counters.factorization_cnt,
                         t, eqfeas(nlp, x), Float64(obj(nlp, x)), rec)
    finally
        cleanup(nlp)
    end
end

# ---- serialization (one trajectory CSV per run + a summary CSV) -----------------------
const _COLS = (:k, :phase, :muB, :muP, :inf_pr, :inf_du, :inf_compl,
               :obj_int, :obj_true, :seq_inf, :alpha, :alpha_z, :ftype,
               :del_w, :del_c, :n_factor, :facts_iter, :ls_l, :restoration_t)

function save_trajectory(res::RunResult, dir)
    mkpath(dir)
    path = joinpath(dir, res.name * ".csv")
    open(path, "w") do io
        println(io, join(_COLS, ","))
        for row in res.rec.rows
            println(io, join((row[c] for c in _COLS), ","))
        end
    end
    return path
end

function save_summary(results::AbstractVector{RunResult}, dir)
    mkpath(dir)
    path = joinpath(dir, "summary.csv")
    open(path, "w") do io
        println(io, "name,status,iters,n_factor,time_s,eqfeas,obj")
        for r in results
            @printf(io, "%s,%s,%d,%d,%.4f,%.6e,%.10e\n",
                    r.name, r.status, r.iters, r.n_factor, r.time, r.eqfeas, r.obj)
        end
    end
    return path
end
