# bcoupled penalty sweep over a CUTEst regime, FP64. Single schedule (no grid):
# CoshKernel + BARRIER-COUPLED μ_P continuation — NO E-gate, NO s-gate; μ_P bumps (linear κ_P=2)
# in lockstep with each barrier reduction (μ_B drop). This is the variant that broke HS7's
# dual-convergence deadlock (MAX_ITER→SUCCEEDED). Condensed KKT, default solver+scaling, tol=1e-8.
#
# Usage: julia --project=bench/penalty bench/penalty_lab/run_penalty_bcoupled.jl [regime] [nworkers]
# Writes results/penalty_bcoupled/<regime>.csv (one row per problem).

using Distributed
const HERE   = @__DIR__
const PROJ   = abspath(joinpath(HERE, "..", "penalty"))
const REGIME = length(ARGS) >= 1 ? ARGS[1] : "le500_eq"
const NW     = min(length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 22, 23)   # shared node: ≤24 cores

probs = String.(filter(!isempty, strip.(readlines(joinpath(HERE, "regimes", "$REGIME.txt")))))
println("BCOUPLED sweep: regime=$REGIME problems=$(length(probs)) workers=$NW (FP64, κ_P=2 linear, μ_B-coupled)")

# ---- Phase 1: serial pre-decode (FP64 libs; cached from earlier diag runs) ----
using MadNLP, CUTEst, NLPModels
const MASTSIF_PATH = ENV["MASTSIF"]
isbuilt(name) = isfile(joinpath(CUTEst.libsif_path, "lib$(name)_double.so"))
println("pre-decoding $(count(!isbuilt, probs)) missing...")
predecode_failed = String[]
for (i,p) in enumerate(probs)
    isbuilt(p) && continue
    try; nlp = CUTEstModel(p); finalize(nlp); catch e; push!(predecode_failed, p); println("  DECODE FAIL $p"); end
end
good = setdiff(probs, predecode_failed)
println("pre-decode done; failed=$(length(predecode_failed)); solving $(length(good))"); flush(stdout)

# ---- Phase 2: parallel solve ----
addprocs(NW; exeflags = "--project=$(PROJ)")
@everywhere ENV["MASTSIF"] = $(MASTSIF_PATH)
@everywhere begin
    using MadNLP, CUTEst, NLPModels
    const MAXWALL = 300.0
    _san(s) = replace(string(s), r"[,\n\r]" => ";")

    # Barrier-coupled linear continuation: bump μ_P iff μ_B was just reduced. No E-gate, no s-gate.
    mutable struct BCoupled <: MadNLP.AbstractPenaltySchedule
        kappa_P::Float64; theta_P::Float64; muP_max::Float64
        n_muP::Int; n_muB::Int
        muP_cur::Base.RefValue{Float64}; last_muB::Base.RefValue{Float64}
    end
    BCoupled(; kappa_P=2.0, theta_P=1.0, muP_max=1.0e7) =
        BCoupled(kappa_P, theta_P, muP_max, 0, 0, Ref(NaN), Ref(NaN))
    function MadNLP.update_penalty!(s::BCoupled, eh::KernelPenaltyEquality,
                                    solver::MadNLP.AbstractMadNLPSolver{T}) where T
        μB = Float64(MadNLP.get_mu(solver))
        if !isnan(s.last_muB[]) && μB < s.last_muB[]          # μ_B dropped this iteration
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
    function solve_one(name::AbstractString)
        local nlp
        try; nlp = CUTEstModel(name; decode=false)
        catch e; return _err(name, -1, -1, "LOAD_ERROR:"*_san(first(split(sprint(showerror,e),'\n')))); end
        sched = BCoupled(kappa_P=2.0, theta_P=1.0, muP_max=1.0e7)
        treat = KernelPenaltyEquality(CoshKernel(); schedule=sched, muP=2.0)
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

println("solving $(length(good)) problems on $NW workers..."); flush(stdout)
results = pmap(solve_one, good;
               on_error = ex -> (:__EXC, string(typeof(ex)), first(split(sprint(showerror, ex), '\n'))))

outdir = joinpath(HERE, "results", "penalty_bcoupled"); mkpath(outdir)
csv = joinpath(outdir, "$(REGIME).csv")
cols = (:name, :nvar, :ncon, :status, :iter, :nfact, :time, :eqfeas, :obj, :muP_final, :n_muP, :n_muB)
open(csv, "w") do io
    println(io, join(cols, ","))
    for p in predecode_failed
        println(io, join((p,-1,-1,"DECODE_ERROR",-1,-1,0.0,NaN,NaN,NaN,-1,-1), ","))
    end
    for (p, res) in zip(good, results)
        row = res isa NamedTuple ? res :
              (name=p,nvar=-1,ncon=-1,status=(res isa Tuple ? _san("CRASH:"*res[3]) : "CRASH"),
               iter=-1,nfact=-1,time=0.0,eqfeas=NaN,obj=NaN,muP_final=NaN,n_muP=-1,n_muB=-1)
        println(io, join((row[c] for c in cols), ","))
    end
end
println("wrote $csv")

rows  = [r for r in results if r isa NamedTuple]
gm(v) = (w=filter(>(0), v); isempty(w) ? NaN : exp(sum(log, w)/length(w)))
succ  = [r for r in rows if r.status=="SOLVE_SUCCEEDED"]
tight = count(r -> r.status=="SOLVE_SUCCEEDED" && r.eqfeas<=1e-8, rows)
println("BCOUPLED $REGIME: $(length(rows)) solves  SUCCEEDED=$(length(succ))  TIGHT(eqfeas≤1e-8)=$tight  "*
        "geomean_nfact(solved)=$(round(gm([Float64(r.nfact) for r in succ]),digits=1))")
println("  (ref: baseline tight=116 ; best linear-gated diag tE10_k2 = 34 tight / 66 succ)")
println("PENALTY_BCOUPLED_DONE")
