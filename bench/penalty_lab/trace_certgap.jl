# Endgame trace for the ALM certification gap: problems that reach tight eqfeas (≤1e-8) but end
# SEARCH_DIR/MAX_ITER instead of SOLVE_SUCCEEDED. Per iteration it records the three KKT residuals,
# the scaled equality slack ‖s‖∞, the multiplier estimate vs the true dual (‖λ−y‖∞), the AL
# slack-row stationarity ‖λ+μ_P·s−y‖∞, the line-search outcome (α, ftype), and the AL counters —
# everything needed to pinpoint why inf_total≤tol ∧ ‖s‖≤tol never co-occur.
# Config = alm_k10 (AugmentedLagrangian κ_P=10, η=0.25, muP0=10, muP_max=1e7), QuadraticKernel, tol=1e-8.
# Usage: julia --project=bench/penalty bench/penalty_lab/trace_certgap.jl [NAME ...]

using MadNLP, CUTEst, NLPModels, LinearAlgebra, Printf
const NAMES = length(ARGS) >= 1 ? ARGS :
    String.(filter(!isempty, strip.(readlines("/tmp/claude-1025/-home-yifan-xu-MadNLP-jl/76efbc10-da4a-42b3-ae87-d69db7eaec00/scratchpad/certgap.txt"))))
const TOL = 1e-8

mutable struct CG <: MadNLP.AbstractUserCallback
    rows::Vector{NamedTuple}
end
CG() = CG(NamedTuple[])
_phase(mode) = Symbol(replace(string(nameof(typeof(mode))), "UserCallback" => ""))
function (rec::CG)(solver, mode)
    cnt = MadNLP.get_cnt(solver)
    eh  = MadNLP.get_cb(solver).equality_handler
    es  = eh.ind_eqslack
    sx  = view(MadNLP.slack(MadNLP.get_x(solver)), es)
    yv  = view(MadNLP.get_y(solver), es)
    lam = eh.lambda
    μ   = Float64(eh.muP[])
    s_inf = isempty(es) ? 0.0 : Float64(norm(sx, Inf))
    y_inf = isempty(es) ? 0.0 : Float64(norm(yv, Inf))
    lam_inf = isempty(es) ? 0.0 : Float64(norm(lam, Inf))
    lamy = isempty(es) ? 0.0 : Float64(maximum(abs.(lam .- yv)))                  # ‖λ − y‖∞ (multiplier convergence)
    stat = isempty(es) ? 0.0 : Float64(maximum(abs.(lam .+ μ .* sx .- yv)))       # ‖λ + μ_P·s − y‖∞ (AL slack stationarity)
    sch = eh.schedule
    push!(rec.rows, (k=cnt.k, phase=_phase(mode), muB=Float64(MadNLP.get_mu(solver)), muP=μ,
        inf_pr=Float64(MadNLP.get_inf_pr(solver)), inf_du=Float64(MadNLP.get_inf_du(solver)),
        inf_compl=Float64(MadNLP.get_inf_compl(solver)), inf_total=Float64(MadNLP.get_inf_total(solver)),
        s_inf=s_inf, y_inf=y_inf, lam_inf=lam_inf, lamy_gap=lamy, stat_resid=stat,
        alpha=Float64(MadNLP.get_alpha(solver)), ftype=string(MadNLP.get_ftype(solver)),
        del_w=Float64(MadNLP.get_del_w(solver)),
        n_lam=sch.n_updates[], n_bump=sch.n_bumps[], froz=(s_inf <= TOL ? 1 : 0)))
    return true
end

function _eqfeas(nlp, x)
    m = NLPModels.get_ncon(nlp); m == 0 && return 0.0
    c = similar(x, m); NLPModels.cons!(nlp, x, c)
    lc = NLPModels.get_lcon(nlp); uc = NLPModels.get_ucon(nlp)
    return Float64(maximum(abs.(c .- lc) .* (lc .== uc)))
end

outdir = joinpath(@__DIR__, "results", "trace_certgap"); mkpath(outdir)
cols = (:k,:phase,:muB,:muP,:inf_pr,:inf_du,:inf_compl,:inf_total,:s_inf,:y_inf,:lam_inf,:lamy_gap,:stat_resid,:alpha,:ftype,:del_w,:n_lam,:n_bump,:froz)
for name in NAMES
    nlp = try CUTEstModel(name; decode=false) catch; continue end
    rec = CG()
    sched = MadNLP.AugmentedLagrangian(kappa_P=10.0, theta_P=1.0, muP_max=1.0e7)  # corrected defaults (eta_lambda=0.5, omega0=1, beta_omega=0.2, stall_patience=6, bump_patience=5)
    treat = KernelPenaltyEquality(QuadraticKernel(); schedule=sched, muP=10.0)
    local r
    try
        r = madnlp(nlp; kkt_system=MadNLP.SparseCondensedKKTSystem, tol=TOL, acceptable_tol=0.0,
                   equality_treatment=treat, intermediate_callback=rec, max_iter=3000, max_wall_time=120.0,
                   print_level=MadNLP.ERROR)
    catch e; @printf("%-10s ERROR %s\n", name, first(split(sprint(showerror,e),'\n'))); finalize(nlp); continue end
    ef = _eqfeas(nlp, r.solution)
    csv = joinpath(outdir, "$(name).csv")
    open(csv,"w") do io
        println(io, join(cols, ","))
        for row in rec.rows; println(io, join((row[c] for c in cols), ",")); end
    end
    @printf("##### %-10s status=%-30s iters=%d eqfeas=%.2e muP=%.0f n_lam=%d n_bump=%d (%d rows) #####\n",
            name, string(r.status), r.iter, ef, sched.muP_cur[], sched.n_updates[], sched.n_bumps[], length(rec.rows))
    # dump the last 18 iterations (the collapse)
    @printf("%5s %8s %9s %9s %9s %9s %9s %9s %9s %4s %9s %4s\n",
            "k","muP","inf_pr","inf_du","inf_cmpl","s_inf","lamy_gap","stat_res","alpha","ft","del_w","froz")
    for row in rec.rows[max(1,end-17):end]
        @printf("%5d %8.0f %9.1e %9.1e %9.1e %9.1e %9.1e %9.1e %9.1e %4s %9.1e %4d\n",
                row.k, row.muP, row.inf_pr, row.inf_du, row.inf_compl, row.s_inf, row.lamy_gap, row.stat_resid, row.alpha, row.ftype, row.del_w, row.froz)
    end
    println()
    finalize(nlp)
end
println("TRACE_CERTGAP_DONE")
