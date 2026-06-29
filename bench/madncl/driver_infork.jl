# In-fork kernel-NCL benchmark: drives MadLP.madncl (the faithful in-MadNLP port) through the
# no-pivot KKT systems on CPU, and compares against madncl_reference/bench_results/.
#   k2r_ldl  : K2rAuglagKKTSystem + LDLSolver     (stabilized augmented, no pivot)
#   k1s_chol : K1sAuglagKKTSystem + CHOLMODSolver (condensed SPD, no pivot)
# Configs (match the gate):
#   quad : NCLOptions defaults (rho_max=1e12, tau_rho=10, predict_resid=false)
#   cosh : rho_max=1e10, tau_rho=6.310, predict_resid=true
#
# Usage: julia --project=bench/madncl bench/madncl/driver_infork.jl [regime] [kernel] [nworkers] [configs]
#   regime  : name under madncl_reference/bench_regimes/  (default le500_eq)
#   kernel  : quad | cosh                                  (default quad)
#   nworkers: capped at 22 (shared node ≤24 cores)         (default 22)
#   configs : k2r | k1s | both                             (default k2r)
# Writes results_infork/<regime>__<kernel>__<cfg>.csv

using Distributed
const HERE    = @__DIR__
const FORK    = abspath(joinpath(HERE, "..", ".."))            # the MadNLP fork root
const PROJ    = HERE                                           # bench/madncl env (devs the fork)
const REGDIR  = joinpath(FORK, "madncl_reference", "bench_regimes")
const REGIME  = length(ARGS) >= 1 ? ARGS[1] : "le500_eq"
const KERNEL  = length(ARGS) >= 2 ? ARGS[2] : "quad"
const NW      = min(length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 22, 22)   # ≤24 cores
const CFGARG  = length(ARGS) >= 4 ? ARGS[4] : "k2r"
const CONFIGS = CFGARG == "both" ? [:k2r_ldl, :k1s_chol] : CFGARG == "k1s" ? [:k1s_chol] : [:k2r_ldl]
@assert KERNEL in ("quad", "cosh") "kernel must be quad|cosh, got $KERNEL"

probs = String.(filter(!isempty, strip.(readlines(joinpath(REGDIR, "$REGIME.txt")))))
println("KERNEL-NCL (in-fork): regime=$REGIME kernel=$KERNEL problems=$(length(probs)) workers=$NW configs=$CONFIGS (FP64, CPU)")

using MadNLP, CUTEst, NLPModels
isbuilt(name) = isfile(joinpath(CUTEst.libsif_path, "lib$(name)_double.so"))
println("pre-decoding $(count(!isbuilt, probs)) missing..."); flush(stdout)
predecode_failed = String[]
for p in probs
    isbuilt(p) && continue
    try; nlp = CUTEstModel(p); finalize(nlp); catch e; push!(predecode_failed, p); end
end
good = setdiff(probs, predecode_failed)
println("pre-decode done; failed=$(length(predecode_failed)); solving $(length(good)) × $(length(CONFIGS))"); flush(stdout)

addprocs(NW; exeflags = "--project=$(PROJ)")
@everywhere begin
    using MadNLP, CUTEst, NLPModels
    const MAXWALL = 100.0
    _san(s) = replace(string(s), r"[,\n\r]" => ";")
    _setup(sym) = sym === :k2r_ldl ? (MadNLP.K2rAuglagKKTSystem, MadNLP.LDLSolver) :
                                      (MadNLP.K1sAuglagKKTSystem, MadNLP.CHOLMODSolver)
    _kernel(k) = k == "cosh" ? MadNLP.CoshKernel() : MadNLP.QuadraticKernel()
    function _opt(k)
        if k == "cosh"
            return MadNLP.NCLOptions{Float64}(opt_tol=1e-8, feas_tol=1e-8, constr_viol_tol=1e-8,
                verbose=false, rho_max=1e10, tau_rho=6.310, predict_resid=true)
        else
            return MadNLP.NCLOptions{Float64}(opt_tol=1e-8, feas_tol=1e-8, constr_viol_tol=1e-8, verbose=false)
        end
    end
    function _eqfeas(nlp, x)
        m = NLPModels.get_ncon(nlp); m == 0 && return 0.0
        c = similar(x, m); NLPModels.cons!(nlp, x, c)
        lc = NLPModels.get_lcon(nlp); uc = NLPModels.get_ucon(nlp)
        return Float64(maximum(abs.(c .- lc) .* (lc .== uc)))
    end
    _err(name, nv, nc, st) = (name=name, nvar=nv, ncon=nc, status=st, iter=-1, nfact=-1,
        time=0.0, eqfeas=NaN, obj=NaN)
    function solve_one(name::AbstractString, sym::Symbol, kname::AbstractString)
        local nlp
        try; nlp = CUTEstModel(name; decode=false)
        catch e; return _err(name, -1, -1, "LOAD_ERROR:"*_san(first(split(sprint(showerror,e),'\n')))); end
        kkt, ls = _setup(sym)
        try
            nvar, ncon = nlp.meta.nvar, nlp.meta.ncon
            t = @elapsed r = MadNLP.madncl(nlp; ncl_options=_opt(kname), kernel=_kernel(kname),
                                    kkt_system=kkt, linear_solver=ls,
                                    max_iter=3000, max_wall_time=MAXWALL, print_level=MadNLP.ERROR)
            return (name=name, nvar=nvar, ncon=ncon, status=string(r.status), iter=r.iter,
                    nfact=r.counters.factorization_cnt, time=t, eqfeas=_eqfeas(nlp, r.solution),
                    obj=Float64(r.objective))
        catch e
            return _err(name, nlp.meta.nvar, nlp.meta.ncon, "ERROR:"*_san(first(split(sprint(showerror,e),'\n'))))
        finally
            finalize(nlp)
        end
    end
end

outdir = joinpath(HERE, "results_infork"); mkpath(outdir)
cols = (:name, :nvar, :ncon, :status, :iter, :nfact, :time, :eqfeas, :obj)
gm(v) = (w=filter(>(0), v); isempty(w) ? NaN : exp(sum(log, w)/length(w)))
for (ci, sym) in enumerate(CONFIGS)
    println("\n[$ci/$(length(CONFIGS))] kernel=$KERNEL $sym (no numerical pivoting)"); flush(stdout)
    results = pmap(p -> solve_one(p, sym, KERNEL), good;
                   on_error = ex -> (:__EXC, string(typeof(ex)), first(split(sprint(showerror,ex),'\n'))))
    csv = joinpath(outdir, "$(REGIME)__$(KERNEL)__$(sym).csv")
    open(csv, "w") do io
        println(io, join(cols, ","))
        for p in predecode_failed
            println(io, join((p,-1,-1,"DECODE_ERROR",-1,-1,0.0,NaN,NaN), ","))
        end
        for (p,res) in zip(good, results)
            row = res isa NamedTuple ? res :
                  (name=p,nvar=-1,ncon=-1,status=(res isa Tuple ? _san("CRASH:"*res[3]) : "CRASH"),
                   iter=-1,nfact=-1,time=0.0,eqfeas=NaN,obj=NaN)
            println(io, join((row[c] for c in cols), ","))
        end
    end
    rows = [r for r in results if r isa NamedTuple]
    succ = [r for r in rows if r.status=="SOLVE_SUCCEEDED"]
    tight = count(r -> r.status=="SOLVE_SUCCEEDED" && r.eqfeas<=1e-8, rows)
    nfail = count(r -> occursin("ERROR", r.status) || occursin("CRASH", r.status), rows)
    gmf = gm([Float64(r.nfact) for r in succ])
    gmi = gm([Float64(r.iter) for r in succ])
    println("  → $sym TIGHT=$tight SUCC=$(length(succ)) solver_fails=$nfail gm_nfact=$(round(gmf,digits=2)) gm_iter=$(round(gmi,digits=2)) → $csv"); flush(stdout)
end
println("\nINFORK_DONE regime=$REGIME kernel=$KERNEL")
