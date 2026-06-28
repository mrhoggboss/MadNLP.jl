# Vanilla MadNLP baseline in QUAD precision (Float128) over a CUTEst regime — DEFAULT OPTIONS,
# CONDENSED KKT system. The quad analog of run_regime_cpu.jl's condensed baseline.
#
# "Default options": equality_treatment defaults to RelaxEquality (options.jl:149 — the kkt-dependent
# default for SparseCondensedKKTSystem), tol/nlp_scaling/max_iter all default. The ONLY forced
# deviations from F64 are (a) Float128 throughout (CUTEstModel{Float128}, quadruple decode) and
# (b) linear_solver = LDLSolver — the sole Float128-capable sparse solver (MUMPS/CHOLMOD are
# double-only). Standing experiment policy: acceptable_tol=0 (no ACCEPTABLE exit), 300s wall cap.
#
# Usage: julia --project=bench/penalty bench/penalty_lab/run_regime_cpu_quad.jl [regime] [nworkers] [tol]
# Writes results/cpu_baseline/<regime>_cond_QUAD_tol<tol>.csv (one row per problem).

using Distributed

const HERE   = @__DIR__
const PROJ   = abspath(joinpath(HERE, "..", "penalty"))
const REGIME = length(ARGS) >= 1 ? ARGS[1] : "le500_eq"
const NW     = min(length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 22, 23)   # shared node: ≤24 cores

using Quadmath
const QT  = Float128
const TOL = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.0e-8       # recorded as F64, passed as QT

probs = String.(filter(!isempty, strip.(readlines(joinpath(HERE, "regimes", "$REGIME.txt")))))
println("QUAD BASELINE (default condensed): regime=$REGIME problems=$(length(probs)) workers=$NW tol=$TOL")

# ---- Phase 1: serial pre-decode in QUADRUPLE (separate cache: libNAME_quadruple.so) ----
using MadNLP, CUTEst, NLPModels
const MASTSIF_PATH = ENV["MASTSIF"]
isbuilt_q(name) = isfile(joinpath(CUTEst.libsif_path, "lib$(name)_quadruple.so"))
println("pre-decoding (quad) $(count(!isbuilt_q, probs)) missing of $(length(probs))...")
predecode_failed = String[]
for (i, p) in enumerate(probs)
    isbuilt_q(p) && continue
    try; nlp = CUTEstModel{QT}(p); finalize(nlp); catch e; push!(predecode_failed, p); println("  DECODE FAIL $p"); end
    i % 25 == 0 && println("  ...$i/$(length(probs))")
end
good = setdiff(probs, predecode_failed)
println("quad pre-decode done; failed=$(length(predecode_failed)); solving $(length(good)) problems"); flush(stdout)

# ---- Phase 2: parallel quad solve, default condensed ----
addprocs(NW; exeflags = "--project=$(PROJ)")
@everywhere ENV["MASTSIF"] = $(MASTSIF_PATH)
@everywhere begin
    using MadNLP, CUTEst, NLPModels, Quadmath
    const QT = Float128
    const MAXWALL = 300.0
    _san(s) = replace(string(s), r"[,\n\r]" => ";")
    function _eqfeas(nlp, x)
        m = NLPModels.get_ncon(nlp); m == 0 && return 0.0
        c = similar(x, m); NLPModels.cons!(nlp, x, c)
        lc = NLPModels.get_lcon(nlp); uc = NLPModels.get_ucon(nlp)
        return Float64(maximum(abs.(c .- lc) .* (lc .== uc)))
    end
    function solve_one(name::AbstractString, tol)
        local nlp
        try; nlp = CUTEstModel{QT}(name; decode = false)
        catch e
            return (name=name, nvar=-1, ncon=-1, status="LOAD_ERROR:"*_san(first(split(sprint(showerror,e),'\n'))),
                    iter=-1, nfact=-1, time=0.0, eqfeas=NaN, obj=NaN, tol=NaN)
        end
        try
            nvar, ncon = nlp.meta.nvar, nlp.meta.ncon
            # default options: equality_treatment auto = RelaxEquality (condensed). Forced for quad:
            # linear_solver = LDLSolver (only Float128-capable). T-typed tol/acceptable/walltime.
            t = @elapsed r = madnlp(nlp; kkt_system = MadNLP.SparseCondensedKKTSystem,
                                    linear_solver = MadNLP.LDLSolver, tol = QT(tol),
                                    acceptable_tol = QT(0.0), max_wall_time = QT(MAXWALL),
                                    print_level = MadNLP.ERROR)
            return (name=name, nvar=nvar, ncon=ncon, status=string(r.status), iter=r.iter,
                    nfact=r.counters.factorization_cnt, time=t, eqfeas=_eqfeas(nlp, r.solution),
                    obj=Float64(r.objective), tol=Float64(r.options.tol))
        catch e
            return (name=name, nvar=nlp.meta.nvar, ncon=nlp.meta.ncon,
                    status="ERROR:"*_san(first(split(sprint(showerror,e),'\n'))),
                    iter=-1, nfact=-1, time=0.0, eqfeas=NaN, obj=NaN, tol=NaN)
        finally
            finalize(nlp)
        end
    end
end

println("solving $(length(good)) problems on $NW workers..."); flush(stdout)
results = pmap(p -> solve_one(p, TOL), good;
               on_error = ex -> (:__EXC, string(typeof(ex)), first(split(sprint(showerror, ex), '\n'))))

outdir = joinpath(HERE, "results", "cpu_baseline"); mkpath(outdir)
csv = joinpath(outdir, "$(REGIME)_cond_QUAD_tol$(TOL).csv")
cols = (:name, :nvar, :ncon, :status, :iter, :nfact, :time, :eqfeas, :obj, :tol)
open(csv, "w") do io
    println(io, join(cols, ","))
    for p in predecode_failed
        println(io, join((p, -1, -1, "DECODE_ERROR", -1, -1, 0.0, NaN, NaN, NaN), ","))
    end
    for (p, res) in zip(good, results)
        st = res isa Tuple && length(res) == 3 && res[1] === :__EXC ? _san("CRASH:" * res[3]) : "CRASH"
        row = res isa NamedTuple ? res :
              (name=p, nvar=-1, ncon=-1, status=st, iter=-1, nfact=-1, time=0.0, eqfeas=NaN, obj=NaN, tol=NaN)
        println(io, join((row[c] for c in cols), ","))
    end
end
println("wrote $csv")

rows  = [r for r in results if r isa NamedTuple]
succ  = count(r -> r.status == "SOLVE_SUCCEEDED", rows)
tight = count(r -> r.status == "SOLVE_SUCCEEDED" && r.eqfeas <= 1e-8, rows)
acc   = count(r -> occursin("ACCEPTABLE", r.status), rows)
println("QUAD condensed baseline: $(length(rows)) solves  SUCCEEDED=$succ  TIGHT(eqfeas≤1e-8)=$tight  ACCEPTABLE=$acc")
println("REGIME_CPU_QUAD_DONE")
