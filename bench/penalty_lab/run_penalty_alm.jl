# Augmented-Lagrangian (method-of-multipliers) sweep on the quadratic kernel, FP64. Phase-1 validation
# of the ALM: does λ→y drive the 62 feasibility-endgame problems to TIGHT (eqfeas≤1e-8) at finite μ_P?
# acceptable_tol=0 (force tight-or-fail; the ‖s‖-termination gate is in-source). Compare to baseline 116
# and quad bcoup_k2 = 55.  Usage: julia --project=bench/penalty bench/penalty_lab/run_penalty_alm.jl [regime] [nworkers]
# Writes results/penalty_alm/<regime>__<config>.csv + summary_alm.csv.

using Distributed
const HERE   = @__DIR__
const PROJ   = abspath(joinpath(HERE, "..", "penalty"))
const REGIME = length(ARGS) >= 1 ? ARGS[1] : "le500_eq"
const NW     = min(length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 22, 23)

# (name, muP0, kw...) for MadNLP.AugmentedLagrangian
const CONFIGS = [
    (name="alm_k10",      muP0=10.0,  kw=(kappa_P=10.0, theta_P=1.0, muP_max=1.0e7)),                    # corrected defaults (eta_lambda=0.5, omega0=1, beta_omega=0.2, stall_patience=6, bump_patience=5)
    (name="alm_mom100",   muP0=100.0, kw=(kappa_P=1.0,  theta_P=1.0, muP_max=100.0)),                    # pure MoM, fixed μ_P=100
    (name="alm_k10_eta25",muP0=10.0,  kw=(kappa_P=10.0, theta_P=1.0, eta_lambda=0.25, muP_max=1.0e7)),   # stricter feasibility-stall ⇒ more μ_P bumps
]

probs = String.(filter(!isempty, strip.(readlines(joinpath(HERE, "regimes", "$REGIME.txt")))))
println("ALM sweep: regime=$REGIME problems=$(length(probs)) workers=$NW configs=$(length(CONFIGS)) (FP64, QuadraticKernel + AugmentedLagrangian)")

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
    using MadNLP, CUTEst, NLPModels
    const MAXWALL = 300.0
    _san(s) = replace(string(s), r"[,\n\r]" => ";")
    function _eqfeas(nlp, x)
        m = NLPModels.get_ncon(nlp); m == 0 && return 0.0
        c = similar(x, m); NLPModels.cons!(nlp, x, c)
        lc = NLPModels.get_lcon(nlp); uc = NLPModels.get_ucon(nlp)
        return Float64(maximum(abs.(c .- lc) .* (lc .== uc)))
    end
    _err(name, nv, nc, st) = (name=name, nvar=nv, ncon=nc, status=st, iter=-1, nfact=-1,
        time=0.0, eqfeas=NaN, obj=NaN, muP_final=NaN, n_lam=-1, n_bumps=-1)
    function solve_one(name::AbstractString, cfg)
        local nlp
        try; nlp = CUTEstModel(name; decode=false)
        catch e; return _err(name, -1, -1, "LOAD_ERROR:"*_san(first(split(sprint(showerror,e),'\n')))); end
        sched = MadNLP.AugmentedLagrangian(; cfg.kw...)
        treat = KernelPenaltyEquality(QuadraticKernel(); schedule=sched, muP=cfg.muP0)
        try
            nvar, ncon = nlp.meta.nvar, nlp.meta.ncon
            t = @elapsed r = madnlp(nlp; kkt_system=MadNLP.SparseCondensedKKTSystem, tol=1e-8,
                                    acceptable_tol=0.0, equality_treatment=treat,
                                    max_wall_time=MAXWALL, print_level=MadNLP.ERROR)
            return (name=name, nvar=nvar, ncon=ncon, status=string(r.status), iter=r.iter,
                    nfact=r.counters.factorization_cnt, time=t, eqfeas=_eqfeas(nlp, r.solution),
                    obj=Float64(r.objective), muP_final=sched.muP_cur[], n_lam=sched.n_updates[], n_bumps=sched.n_bumps[])
        catch e
            r = _err(name, nlp.meta.nvar, nlp.meta.ncon, "ERROR:"*_san(first(split(sprint(showerror,e),'\n'))))
            return merge(r, (muP_final=sched.muP_cur[], n_lam=sched.n_updates[], n_bumps=sched.n_bumps[]))
        finally
            finalize(nlp)
        end
    end
end

outdir = joinpath(HERE, "results", "penalty_alm"); mkpath(outdir)
cols = (:name,:nvar,:ncon,:status,:iter,:nfact,:time,:eqfeas,:obj,:muP_final,:n_lam,:n_bumps)
gm(v) = (w=filter(>(0), v); isempty(w) ? NaN : exp(sum(log, w)/length(w)))
med(v) = (w=sort(v); isempty(w) ? NaN : w[cld(length(w),2)])
summ = NamedTuple[]
for (ci,c) in enumerate(CONFIGS)
    println("\n[$ci/$(length(CONFIGS))] $(c.name)  muP0=$(c.muP0)  $(c.kw)"); flush(stdout)
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
                   iter=-1,nfact=-1,time=0.0,eqfeas=NaN,obj=NaN,muP_final=NaN,n_lam=-1,n_bumps=-1)
            println(io, join((row[col] for col in cols), ","))
        end
    end
    rows = [r for r in results if r isa NamedTuple]
    succ = [r for r in rows if r.status=="SOLVE_SUCCEEDED"]
    tight = count(r -> r.status=="SOLVE_SUCCEEDED" && r.eqfeas<=1e-8, rows)
    feas_any = count(r -> isfinite(r.eqfeas) && r.eqfeas<=1e-8, rows)   # eqfeas tight regardless of status
    gmf = gm([Float64(r.nfact) for r in succ]); mlam = med([r.n_lam for r in rows if r.n_lam>=0])
    push!(summ, (name=c.name, n=length(rows), tight=tight, succ=length(succ), feas_any=feas_any, gm_nfact=gmf, med_lam=mlam))
    println("  → TIGHT=$tight SUCC=$(length(succ)) eqfeas≤1e-8(any status)=$feas_any geomean_nfact=$(round(gmf,digits=1)) med_n_lam=$mlam"); flush(stdout)
    open(joinpath(outdir, "summary_alm.csv"), "w") do io
        println(io, "config,n,tight,succeeded,eqfeas_tight_anystatus,geomean_nfact_solved,median_n_lambda")
        for s in summ; println(io, join((s.name,s.n,s.tight,s.succ,s.feas_any,round(s.gm_nfact,digits=2),s.med_lam), ",")); end
    end
end
println("\n(refs: baseline 116 ; quad bcoup_k2 = 55 tight / 117 succ ; the 62 feasibility-endgame bucket is the target)")
println("PENALTY_ALM_DONE")
