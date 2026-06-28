# QUADRATIC-kernel pure-penalty sweep, FP64 — the experiment we never ran. Tests "path 1": does
# swapping the cosh kernel for a quadratic (φ=(μ/2)s², D_s=μ constant, no overflow) recover the gap
# to the method of multipliers, while keeping the single-continuous-solve? Same schedules as the
# cosh runs. muP_max raised to 1e9 (quadratic feasibility decays only as s~y/μ_P, so it needs MUCH
# higher μ_P than cosh's s~y/μ_P² to reach tight 1e-8).
#
# Usage: julia --project=bench/penalty bench/penalty_lab/run_penalty_quadratic.jl [regime] [nworkers]
# Writes results/penalty_quadratic/<regime>__<config>.csv + summary_quadratic.csv.

using Distributed
const HERE   = @__DIR__
const PROJ   = abspath(joinpath(HERE, "..", "penalty"))
const REGIME = length(ARGS) >= 1 ? ARGS[1] : "le500_eq"
const NW     = min(length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 22, 23)

const CONFIGS = [
    (name="bcoup_k2",   kind=:bcoup, tau_P=0.0,  kappa_P=2.0,  theta_P=1.0, muP0=2.0,  muP_max=1.0e9),
    (name="bcoup_k10",  kind=:bcoup, tau_P=0.0,  kappa_P=10.0, theta_P=1.0, muP0=10.0, muP_max=1.0e9),
    (name="sgate_tP10", kind=:sgate, tau_P=10.0, kappa_P=2.0,  theta_P=1.0, muP0=2.0,  muP_max=1.0e9),
]

probs = String.(filter(!isempty, strip.(readlines(joinpath(HERE, "regimes", "$REGIME.txt")))))
println("QUADRATIC-kernel penalty: regime=$REGIME problems=$(length(probs)) workers=$NW configs=$(length(CONFIGS)) (FP64, muP_max=1e9)")

using MadNLP, CUTEst, NLPModels
const MASTSIF_PATH = ENV["MASTSIF"]
isbuilt(name) = isfile(joinpath(CUTEst.libsif_path, "lib$(name)_double.so"))
println("pre-decoding $(count(!isbuilt, probs)) missing...")
predecode_failed = String[]
for p in probs
    isbuilt(p) && continue
    try; nlp = CUTEstModel(p); finalize(nlp); catch e; push!(predecode_failed, p); end
end
good = setdiff(probs, predecode_failed)
println("pre-decode done; failed=$(length(predecode_failed)); solving $(length(good)) × $(length(CONFIGS))"); flush(stdout)

addprocs(NW; exeflags = "--project=$(PROJ)")
@everywhere ENV["MASTSIF"] = $(MASTSIF_PATH)
@everywhere begin
    using MadNLP, CUTEst, NLPModels, LinearAlgebra
    const MAXWALL = 100.0
    _san(s) = replace(string(s), r"[,\n\r]" => ";")

    mutable struct SGateOnly <: MadNLP.AbstractPenaltySchedule
        kappa_P::Float64; theta_P::Float64; tau_P::Float64; muP_max::Float64
        n_muP::Int; n_muB::Int
        muP_cur::Base.RefValue{Float64}; last_muB::Base.RefValue{Float64}
    end
    SGateOnly(; kappa_P, theta_P=1.0, tau_P, muP_max=1.0e9) =
        SGateOnly(kappa_P, theta_P, tau_P, muP_max, 0, 0, Ref(NaN), Ref(NaN))
    function MadNLP.update_penalty!(s::SGateOnly, eh::KernelPenaltyEquality,
                                    solver::MadNLP.AbstractMadNLPSolver{T}) where T
        μB = Float64(MadNLP.get_mu(solver))
        (!isnan(s.last_muB[]) && μB < s.last_muB[]) && (s.n_muB += 1)
        s.last_muB[] = μB
        μ = eh.muP[]
        if μ < s.muP_max
            μ_next = min(T(s.muP_max), max(T(s.kappa_P)*μ, μ^T(s.theta_P)))
            sE = isempty(eh.ind_eqslack) ? zero(T) :
                 norm(view(MadNLP.slack(MadNLP.get_x(solver)), eh.ind_eqslack), Inf)
            if sE <= T(s.tau_P) / μ_next
                eh.muP[] = μ_next; s.n_muP += 1
            end
        end
        s.muP_cur[] = Float64(eh.muP[])
        return
    end

    mutable struct BCoupled <: MadNLP.AbstractPenaltySchedule
        kappa_P::Float64; theta_P::Float64; muP_max::Float64
        n_muP::Int; n_muB::Int
        muP_cur::Base.RefValue{Float64}; last_muB::Base.RefValue{Float64}
    end
    BCoupled(; kappa_P, theta_P=1.0, muP_max=1.0e9) =
        BCoupled(kappa_P, theta_P, muP_max, 0, 0, Ref(NaN), Ref(NaN))
    function MadNLP.update_penalty!(s::BCoupled, eh::KernelPenaltyEquality,
                                    solver::MadNLP.AbstractMadNLPSolver{T}) where T
        μB = Float64(MadNLP.get_mu(solver))
        if !isnan(s.last_muB[]) && μB < s.last_muB[]
            s.n_muB += 1
            if eh.muP[] < s.muP_max
                eh.muP[] = min(T(s.muP_max), max(T(s.kappa_P)*eh.muP[], eh.muP[]^T(s.theta_P)))
                s.n_muP += 1
            end
        end
        s.last_muB[] = μB
        s.muP_cur[]  = Float64(eh.muP[])
        return
    end

    function _eqfeas(nlp, x)
        m = NLPModels.get_ncon(nlp); m == 0 && return 0.0
        c = similar(x, m); NLPModels.cons!(nlp, x, c)
        lc = NLPModels.get_lcon(nlp); uc = NLPModels.get_ucon(nlp)
        return Float64(maximum(abs.(c .- lc) .* (lc .== uc)))
    end
    _err(name, nv, nc, st) = (name=name, nvar=nv, ncon=nc, status=st, iter=-1, nfact=-1,
        time=0.0, eqfeas=NaN, obj=NaN, muP_final=NaN, n_muP=-1, n_muB=-1)
    function solve_one(name::AbstractString, cfg)
        local nlp
        try; nlp = CUTEstModel(name; decode=false)
        catch e; return _err(name, -1, -1, "LOAD_ERROR:"*_san(first(split(sprint(showerror,e),'\n')))); end
        sched = cfg.kind === :sgate ?
            SGateOnly(kappa_P=cfg.kappa_P, theta_P=cfg.theta_P, tau_P=cfg.tau_P, muP_max=cfg.muP_max) :
            BCoupled(kappa_P=cfg.kappa_P, theta_P=cfg.theta_P, muP_max=cfg.muP_max)
        treat = KernelPenaltyEquality(QuadraticKernel(); schedule=sched, muP=cfg.muP0)
        try
            nvar, ncon = nlp.meta.nvar, nlp.meta.ncon
            t = @elapsed r = madnlp(nlp; kkt_system=MadNLP.SparseCondensedKKTSystem, tol=1e-8,
                                    acceptable_tol=1e-8, equality_treatment=treat,
                                    max_wall_time=MAXWALL, print_level=MadNLP.ERROR)
            return (name=name, nvar=nvar, ncon=ncon, status=string(r.status), iter=r.iter,
                    nfact=r.counters.factorization_cnt, time=t, eqfeas=_eqfeas(nlp, r.solution),
                    obj=Float64(r.objective), muP_final=sched.muP_cur[], n_muP=sched.n_muP, n_muB=sched.n_muB)
        catch e
            r = _err(name, nlp.meta.nvar, nlp.meta.ncon, "ERROR:"*_san(first(split(sprint(showerror,e),'\n'))))
            return merge(r, (muP_final=sched.muP_cur[], n_muP=sched.n_muP, n_muB=sched.n_muB))
        finally
            finalize(nlp)
        end
    end
end

outdir = joinpath(HERE, "results", "penalty_quadratic"); mkpath(outdir)
cols = (:name,:nvar,:ncon,:status,:iter,:nfact,:time,:eqfeas,:obj,:muP_final,:n_muP,:n_muB)
gm(v) = (w=filter(>(0), v); isempty(w) ? NaN : exp(sum(log, w)/length(w)))
med(v) = (w=sort(v); isempty(w) ? NaN : w[cld(length(w),2)])
summ = NamedTuple[]
for (ci,c) in enumerate(CONFIGS)
    println("\n[$ci/$(length(CONFIGS))] QuadraticKernel $(c.name)  kind=$(c.kind) τ_P=$(c.tau_P) κ_P=$(c.kappa_P) muP0=$(c.muP0) muP_max=$(c.muP_max)"); flush(stdout)
    results = pmap(p -> solve_one(p, c), good;
                   on_error = ex -> (:__EXC, string(typeof(ex)), first(split(sprint(showerror,ex),'\n'))))
    csv = joinpath(outdir, "$(REGIME)__$(c.name).csv")
    open(csv, "w") do io
        println(io, join(cols, ","))
        for p in predecode_failed
            println(io, join((p,-1,-1,"DECODE_ERROR",-1,-1,0.0,NaN,NaN,NaN,-1,-1), ","))
        end
        for (p,res) in zip(good, results)
            row = res isa NamedTuple ? res :
                  (name=p,nvar=-1,ncon=-1,status=(res isa Tuple ? _san("CRASH:"*res[3]) : "CRASH"),
                   iter=-1,nfact=-1,time=0.0,eqfeas=NaN,obj=NaN,muP_final=NaN,n_muP=-1,n_muB=-1)
            println(io, join((row[col] for col in cols), ","))
        end
    end
    rows = [r for r in results if r isa NamedTuple]
    succ = [r for r in rows if r.status=="SOLVE_SUCCEEDED"]
    tight = count(r -> r.status=="SOLVE_SUCCEEDED" && r.eqfeas<=1e-8, rows)
    inv = count(r -> occursin("INVALID_NUMBER", r.status), rows)
    gmf = gm([Float64(r.nfact) for r in succ]); mpf = med([r.muP_final for r in rows if r.muP_final>0])
    push!(summ, (name=c.name, n=length(rows), tight=tight, succ=length(succ), invalid=inv, gm_nfact=gmf, med_muPf=mpf))
    println("  → $(c.name) TIGHT=$tight SUCC=$(length(succ)) INVALID=$inv geomean_nfact=$(round(gmf,digits=1)) med_muP_final=$(round(mpf,digits=0))"); flush(stdout)
    open(joinpath(outdir, "summary_quadratic.csv"), "w") do io
        println(io, "config,n,tight,succeeded,invalid,geomean_nfact_solved,median_muP_final")
        for s in summ; println(io, join((s.name,s.n,s.tight,s.succ,s.invalid,round(s.gm_nfact,digits=2),round(s.med_muPf,digits=1)), ",")); end
    end
end
println("\n(refs: cosh-penalty best=53 ; normcosh aggressive 44/41 ; baseline 116 ; MadNCL quad-NCL 182)")
println("PENALTY_QUADRATIC_DONE")
