# QUAD-PRECISION diagnostic sweep — same 36 configs as run_penalty_diag.jl, in Float128.
# Differences vs the Float64 driver: T=Float128 everywhere; CUTEstModel{Float128} (quadruple decode,
# separate libsif cache); linear_solver=LDLSolver (MUMPS is double-only, LDLFactorizations is the
# Float128-capable sparse solver); T-typed options (set_options! won't convert Float64→Float128);
# OVF = log(floatmax(Float128)) ≈ 11355 (precision-appropriate "overflow_thres" for τ_s).
#
# Grid (36): τ_E ∈ {10,50,100,1000} × κ_P ∈ {2,5,10} (linear) × τ_s ∈ {1, 10, OVF≈11355}.
# Records μ_P/μ_B bumps + the E/s gate-binding split, like the F64 run. s=0 init, ACCEPTABLE off.
# Usage: julia --project=bench/penalty bench/penalty_lab/run_penalty_diag_quad.jl [regime] [nworkers]

using Distributed
const HERE   = @__DIR__
const PROJ   = abspath(joinpath(HERE, "..", "penalty"))
const REGIME = length(ARGS) >= 1 ? ARGS[1] : "le500_eq"
const NW     = min(length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 22, 23)

using Quadmath
const QT  = Float128
const OVF = Float64(log(floatmax(QT)))           # ≈ 11356.5; Float64 so configs stay serializable

const TAU_E  = [10.0, 50.0, 100.0, 1000.0]
const KAPPA  = [2.0, 5.0, 10.0]
const TAU_S  = [1.0, 10.0, OVF]
tsname(ts) = ts == OVF ? "ovf" : string(Int(ts))
const CONFIGS = [ (name = "tE$(Int(tE))_k$(Int(k))_ts$(tsname(ts))",
                   cfg  = (tau_E=tE, kappa_P=k, tau_s=ts, muP0=2.0, muP_max=1.0e7, tol=1.0e-8))
                  for tE in TAU_E for k in KAPPA for ts in TAU_S ]

probs = String.(filter(!isempty, strip.(readlines(joinpath(HERE, "regimes", "$REGIME.txt")))))
println("QUAD DIAG: regime=$REGIME problems=$(length(probs)) workers=$NW configs=$(length(CONFIGS)) (OVF=$(round(OVF,digits=1)))")

# ---- Phase 1: serial pre-decode in QUADRUPLE (separate cache: libNAME_quadruple.so) ----
using MadNLP, CUTEst, NLPModels
const MASTSIF_PATH = ENV["MASTSIF"]
isbuilt_q(name) = isfile(joinpath(CUTEst.libsif_path, "lib$(name)_quadruple.so"))
println("pre-decoding (quad) $(count(!isbuilt_q, probs)) missing...")
predecode_failed = String[]
for (i,p) in enumerate(probs)
    isbuilt_q(p) && continue
    try; nlp = CUTEstModel{QT}(p); finalize(nlp); catch e; push!(predecode_failed, p); println("  DECODE FAIL $p"); end
    i % 25 == 0 && println("  ...$i/$(length(probs))")
end
good = setdiff(probs, predecode_failed)
println("quad pre-decode done; failed=$(length(predecode_failed)); solving $(length(good)) × $(length(CONFIGS))")

# ---- Phase 2: parallel quad solve ----
addprocs(NW; exeflags = "--project=$(PROJ)")
@everywhere ENV["MASTSIF"] = $(MASTSIF_PATH)
@everywhere begin
    using MadNLP, CUTEst, NLPModels, LinearAlgebra, Quadmath
    const QT = Float128
    const MAXWALL = 300.0
    _san(s) = replace(string(s), r"[,\n\r]" => ";")

    mutable struct DiagStatic <: MadNLP.AbstractPenaltySchedule
        kappa_P::Float64; theta_P::Float64; opt_tol_coef::Float64; opt_tol_exp::Float64
        s_thresh::Float64; muP_max::Float64
        n_bump::Int; n_Eblock::Int; n_sblock::Int; n_both::Int; n_muB::Int
        muP_cur::Base.RefValue{Float64}; last_muB::Base.RefValue{Float64}
    end
    DiagStatic(; kappa_P, opt_tol_coef, s_thresh, theta_P=1.0, opt_tol_exp=-1.0, muP_max=Inf) =
        DiagStatic(kappa_P, theta_P, opt_tol_coef, opt_tol_exp, s_thresh, muP_max, 0,0,0,0,0, Ref(NaN), Ref(NaN))
    function MadNLP.update_penalty!(s::DiagStatic, eh::KernelPenaltyEquality, solver::MadNLP.AbstractMadNLPSolver{T}) where T
        μB = Float64(MadNLP.get_mu(solver))
        (!isnan(s.last_muB[]) && μB != s.last_muB[]) && (s.n_muB += 1)
        s.last_muB[] = μB
        μ = eh.muP[]
        if μ < s.muP_max
            μ_next = min(T(s.muP_max), max(T(s.kappa_P)*μ, μ^T(s.theta_P)))
            E  = MadNLP.get_inf_barrier(solver)
            sE = isempty(eh.ind_eqslack) ? zero(T) :
                 norm(view(MadNLP.slack(MadNLP.get_x(solver)), eh.ind_eqslack), Inf)
            Eok = E  <= T(s.opt_tol_coef) * μ^T(s.opt_tol_exp)
            sok = sE <= T(s.s_thresh) / μ_next
            if     Eok && sok;  eh.muP[] = μ_next; s.n_bump   += 1
            elseif Eok && !sok;                    s.n_sblock += 1
            elseif !Eok && sok;                    s.n_Eblock += 1
            else;                                  s.n_both   += 1
            end
        end
        s.muP_cur[] = Float64(eh.muP[])          # convert Float128 → Float64 for recording
        return
    end

    function _eqfeas(nlp, x)
        m = NLPModels.get_ncon(nlp); m == 0 && return 0.0
        c = similar(x, m); NLPModels.cons!(nlp, x, c)
        lc = NLPModels.get_lcon(nlp); uc = NLPModels.get_ucon(nlp)
        return Float64(maximum(abs.(c .- lc) .* (lc .== uc)))
    end
    _err(name, nv, nc, st) = (name=name, nvar=nv, ncon=nc, status=st, iter=-1, nfact=-1, time=0.0,
        eqfeas=NaN, obj=NaN, muP_final=NaN, n_muP=-1, n_muB=-1, Eblock=-1, sblock=-1, both=-1)
    function solve_one(name::AbstractString, cfg)
        local nlp
        try; nlp = CUTEstModel{QT}(name; decode=false)
        catch e; return _err(name, -1, -1, "LOAD_ERROR:"*_san(first(split(sprint(showerror,e),'\n')))); end
        sched = DiagStatic(kappa_P=cfg.kappa_P, opt_tol_coef=cfg.tau_E, s_thresh=cfg.tau_s, muP_max=cfg.muP_max)
        treat = KernelPenaltyEquality(CoshKernel(); schedule=sched, muP=QT(cfg.muP0))
        try
            nvar, ncon = nlp.meta.nvar, nlp.meta.ncon
            t = @elapsed r = madnlp(nlp; kkt_system=MadNLP.SparseCondensedKKTSystem, linear_solver=MadNLP.LDLSolver,
                                    tol=QT(cfg.tol), acceptable_tol=QT(cfg.tol), max_wall_time=QT(MAXWALL),
                                    equality_treatment=treat, print_level=MadNLP.ERROR)
            return (name=name, nvar=nvar, ncon=ncon, status=string(r.status), iter=r.iter, nfact=r.counters.factorization_cnt,
                    time=t, eqfeas=_eqfeas(nlp, r.solution), obj=Float64(r.objective), muP_final=sched.muP_cur[],
                    n_muP=sched.n_bump, n_muB=sched.n_muB, Eblock=sched.n_Eblock, sblock=sched.n_sblock, both=sched.n_both)
        catch e
            r = _err(name, nlp.meta.nvar, nlp.meta.ncon, "ERROR:"*_san(first(split(sprint(showerror,e),'\n'))))
            return merge(r, (muP_final=sched.muP_cur[], n_muP=sched.n_bump, n_muB=sched.n_muB,
                             Eblock=sched.n_Eblock, sblock=sched.n_sblock, both=sched.n_both))
        finally
            finalize(nlp)
        end
    end
end

outdir = joinpath(HERE, "results", "penalty_diag_quad"); mkpath(outdir)
cols = (:name,:nvar,:ncon,:status,:iter,:nfact,:time,:eqfeas,:obj,:muP_final,:n_muP,:n_muB,:Eblock,:sblock,:both)
summ = NamedTuple[]
for (ci, c) in enumerate(CONFIGS)
    println("\n[$ci/$(length(CONFIGS))] $(c.name)  τ_E=$(c.cfg.tau_E) κ=$(c.cfg.kappa_P) τ_s=$(round(c.cfg.tau_s,digits=1))")
    flush(stdout)
    results = pmap(p -> solve_one(p, c.cfg), good; on_error = ex -> (:__EXC, string(typeof(ex)), first(split(sprint(showerror,ex),'\n'))))
    csv = joinpath(outdir, "$(REGIME)__$(c.name).csv")
    open(csv, "w") do io
        println(io, join(cols, ","))
        for p in predecode_failed
            println(io, join((p,-1,-1,"DECODE_ERROR",-1,-1,0.0,NaN,NaN,NaN,-1,-1,-1,-1,-1), ","))
        end
        for (p, res) in zip(good, results)
            row = res isa NamedTuple ? res :
                  (name=p,nvar=-1,ncon=-1,status=(res isa Tuple ? _san("CRASH:"*res[3]) : "CRASH"),
                   iter=-1,nfact=-1,time=0.0,eqfeas=NaN,obj=NaN,muP_final=NaN,n_muP=-1,n_muB=-1,Eblock=-1,sblock=-1,both=-1)
            println(io, join((row[col] for col in cols), ","))
        end
    end
    rows = [r for r in results if r isa NamedTuple]
    tight = count(r -> r.status=="SOLVE_SUCCEEDED" && r.eqfeas<=1e-8, rows)
    succ  = count(r -> r.status=="SOLVE_SUCCEEDED", rows)
    sE = sum(r.Eblock for r in rows if r.Eblock>=0; init=0); ss = sum(r.sblock for r in rows if r.sblock>=0; init=0)
    sb = sum(r.both   for r in rows if r.both>=0;   init=0); smuP = sum(r.n_muP for r in rows if r.n_muP>=0; init=0)
    smuB = sum(r.n_muB for r in rows if r.n_muB>=0; init=0)
    push!(summ, (name=c.name, tau_E=c.cfg.tau_E, kappa_P=c.cfg.kappa_P, tau_s=c.cfg.tau_s, n=length(rows),
                 tight=tight, succ=succ, sum_muP=smuP, sum_muB=smuB, Eblock=sE, sblock=ss, both=sb))
    nb = sE+ss+sb
    println("  → TIGHT=$tight SUCC=$succ | μP_bumps=$smuP μB_bumps=$smuB | gate-block E=$sE s=$ss both=$sb (E-share $(nb>0 ? round(Int,100sE/nb) : 0)%)"); flush(stdout)
    open(joinpath(outdir, "summary_diag_quad.csv"), "w") do io
        println(io, "regime,config,tau_E,kappa_P,tau_s,n,tight,succeeded,sum_muP_bumps,sum_muB_bumps,Eblock,sblock,bothblock")
        for s in summ
            println(io, join((REGIME, s.name, s.tau_E, s.kappa_P, round(s.tau_s,digits=1), s.n, s.tight, s.succ,
                              s.sum_muP, s.sum_muB, s.Eblock, s.sblock, s.both), ","))
        end
    end
end
println("\nwrote $(joinpath(outdir, "summary_diag_quad.csv"))")
println("PENALTY_DIAG_QUAD_DONE")
