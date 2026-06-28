# Diagnostic: track ‖d_s‖∞ (the equality-slack block of the raw Newton direction, BEFORE α) per
# iteration, alongside ‖s‖∞ and μ_P, to see the order of magnitude of d_s and the overflow-safe
# step α_safe ≈ 709/(μ_P·‖d_s‖∞). Config = sgate_tP10 (s-gate only, τ_P=10, κ_P=2, muP0=2),
# normalized-cosh kernel, condensed, default solver+scaling, tol=1e-8. (Uses the CURRENT source
# overflow policy — i.e. +Inf-reject.)
# Usage: julia --project=bench/penalty bench/penalty_lab/trace_ds.jl [NAME1 NAME2 ...]

using MadNLP, CUTEst, NLPModels, LinearAlgebra, Printf
const NAMES = length(ARGS) >= 1 ? ARGS : ["BYRDSPHR", "MARATOS", "HS27"]
const OVF = log(floatmax(Float64))   # ≈ 709.78, cosh-argument overflow threshold

mutable struct SGateOnly <: MadNLP.AbstractPenaltySchedule
    kappa_P::Float64; theta_P::Float64; tau_P::Float64; muP_max::Float64
    muP_cur::Base.RefValue{Float64}
end
SGateOnly(; kappa_P, theta_P=1.0, tau_P, muP_max=1.0e7) = SGateOnly(kappa_P, theta_P, tau_P, muP_max, Ref(NaN))
function MadNLP.update_penalty!(s::SGateOnly, eh::KernelPenaltyEquality, solver::MadNLP.AbstractMadNLPSolver{T}) where T
    μ = eh.muP[]
    if μ < s.muP_max
        μ_next = min(T(s.muP_max), max(T(s.kappa_P)*μ, μ^T(s.theta_P)))
        sE = isempty(eh.ind_eqslack) ? zero(T) : norm(view(MadNLP.slack(MadNLP.get_x(solver)), eh.ind_eqslack), Inf)
        (sE <= T(s.tau_P) / μ_next) && (eh.muP[] = μ_next)
    end
    s.muP_cur[] = Float64(eh.muP[])
    return
end

mutable struct DsRec <: MadNLP.AbstractUserCallback
    rows::Vector{NamedTuple}
end
DsRec() = DsRec(NamedTuple[])
function (rec::DsRec)(solver, mode)
    cnt = MadNLP.get_cnt(solver)
    eh  = MadNLP.get_cb(solver).equality_handler
    es  = eh.ind_eqslack
    sx  = MadNLP.slack(MadNLP.get_x(solver))
    n   = length(MadNLP.variable(MadNLP.get_x(solver)))         # #variables; slacks follow at n+1:n+m
    dfull = MadNLP.primal(MadNLP.get_d(solver))                 # raw primal direction [d_x; d_s], length n+m
    seq = isempty(es) ? 0.0 : Float64(norm(view(sx, es), Inf))
    dsi = isempty(es) ? 0.0 : Float64(norm(view(dfull, n .+ es), Inf))  # ‖d_s‖∞ over eq slacks
    muP = Float64(eh.muP[])
    push!(rec.rows, (k=cnt.k, muP=muP, muB=Float64(MadNLP.get_mu(solver)),
        seq_inf=seq, ds_inf=dsi, alpha=Float64(MadNLP.get_alpha(solver)),
        ftype=string(MadNLP.get_ftype(solver)),
        inf_du=Float64(MadNLP.get_inf_du(solver)), del_w=Float64(MadNLP.get_del_w(solver))))
    return true
end

for name in NAMES
    nlp = try CUTEstModel(name; decode=false) catch; CUTEstModel(name) end
    rec = DsRec()
    sched = SGateOnly(kappa_P=2.0, theta_P=1.0, tau_P=10.0)
    treat = KernelPenaltyEquality(CoshNormalizedKernel(); schedule=sched, muP=2.0)
    r = madnlp(nlp; kkt_system=MadNLP.SparseCondensedKKTSystem, tol=1e-8, acceptable_tol=1e-8,
               equality_treatment=treat, intermediate_callback=rec, max_iter=3000, print_level=MadNLP.ERROR)
    @printf("\n##### %s (nvar=%d ncon=%d): status=%s iters=%d muP_final=%g #####\n",
            name, nlp.meta.nvar, nlp.meta.ncon, string(r.status), r.iter, rec.rows[end].muP)
    @printf("%5s %9s %9s %11s %11s %9s %4s %11s %10s\n",
            "k","muP","seq_inf","ds_inf","muP*ds_inf","alpha","ft","a_safe≈","inf_du")
    # a_safe ≈ (OVF/muP - seq) / ds  (largest α keeping muP*(s+α ds) ≤ OVF); ~OVF/(muP*ds) when s small
    function show(rw)
        asafe = rw.ds_inf > 0 ? max(0.0, (OVF/rw.muP - rw.seq_inf)) / rw.ds_inf : Inf
        @printf("%5d %9.1f %9.1e %11.2e %11.2e %9.1e %4s %11.1e %10.1e\n",
                rw.k, rw.muP, rw.seq_inf, rw.ds_inf, rw.muP*rw.ds_inf, rw.alpha, rw.ftype, asafe, rw.inf_du)
    end
    rows = rec.rows
    idx = sort(unique(vcat(1:min(12,length(rows)), length(rows)-3:length(rows))))
    idx = filter(i -> 1 <= i <= length(rows), idx)
    for i in idx; show(rows[i]); end
    @printf("  ds_inf over run: median=%.2e max=%.2e   muP*ds_inf max=%.2e (OVF=%.0f)\n",
            sort([rw.ds_inf for rw in rows])[cld(length(rows),2)],
            maximum(rw.ds_inf for rw in rows), maximum(rw.muP*rw.ds_inf for rw in rows), OVF)
    finalize(nlp)
end
println("TRACE_DS_DONE")
