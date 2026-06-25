# Vanilla MadNLP baseline over a CUTEst regime — DEFAULT OPTIONS, both CPU KKT systems
# (SparseKKTSystem and SparseCondensedKKTSystem).
#
# "Default options": only kkt_system is set; linear_solver (MumpsSolver), tol (1e-8 augmented /
# 1e-4 condensed), equality_treatment (EnforceEquality / RelaxEquality), nlp_scaling (true),
# max_iter (3000) all take MadNLP's defaults. The ONLY deviation is a per-problem wall-time cap
# (operational safeguard so one problem can't hang the batch).
#
# Parallelism: CUTEst's sifdecoder cd's into a SINGLE shared libsif dir and writes there, so
# concurrent decodes race (segfault). We therefore (1) pre-decode all problems SERIALLY on the
# master to warm the shared cache, then (2) solve in PARALLEL with `CUTEstModel(...; decode=false)`,
# which only dlopens the cached .so (read-only → race-free).
#
# Usage: julia --project=bench/penalty bench/penalty_lab/run_regime_cpu.jl [regime] [nworkers]
#   regime default le5000_eq    nworkers default 24
# Writes results/cpu_baseline/<regime>.csv  (one row per problem × KKT system).

using Distributed

const HERE   = @__DIR__
const PROJ   = abspath(joinpath(HERE, "..", "penalty"))
const REGIME = length(ARGS) >= 1 ? ARGS[1] : "le5000_eq"
# Shared compute node: never use more than 24 cores. Default 22 workers (+ near-idle master ≤ 24);
# clamp any larger request to 23 so workers + master stay within 24.
const NW     = min(length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 22, 23)

probs = String.(filter(!isempty, strip.(readlines(joinpath(HERE, "regimes", "$REGIME.txt")))))  # String, not SubString
println("regime=$REGIME  problems=$(length(probs))  workers=$NW")

# ---- Phase 1: serial pre-decode (warm the shared cache; concurrent decode would race) ----
using MadNLP, CUTEst, NLPModels
const MASTSIF_PATH = ENV["MASTSIF"]   # CUTEst sets this at load; propagate the same to workers
isbuilt(name) = isfile(joinpath(CUTEst.libsif_path, "lib$(name)_double.so"))
println("pre-decoding $(count(!isbuilt, probs)) missing of $(length(probs)) (serial)...")
predecode_failed = String[]
for (i, p) in enumerate(probs)
    isbuilt(p) && continue
    try
        nlp = CUTEstModel(p); finalize(nlp)
    catch e
        push!(predecode_failed, p)
        println("  DECODE FAIL: $p  ($(first(split(sprint(showerror, e), '\n'))))")
    end
    i % 25 == 0 && println("  ...$i/$(length(probs))")
end
good = setdiff(probs, predecode_failed)
println("pre-decode complete; failed=$(length(predecode_failed)); solving $(length(good)) problems")

# ---- Phase 2: parallel solve (decode=false → read-only load, race-free) ----
addprocs(NW; exeflags = "--project=$(PROJ)")
@everywhere ENV["MASTSIF"] = $(MASTSIF_PATH)   # set BEFORE `using CUTEst` so workers see the full collection
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
    function solve_one(name::AbstractString, kkt)
        local nlp
        try
            nlp = CUTEstModel(name; decode = false)
        catch e
            return (name=name, kkt=string(nameof(kkt)), nvar=-1, ncon=-1,
                    status="LOAD_ERROR:" * _san(first(split(sprint(showerror, e), '\n'))),
                    iter=-1, nfact=-1, time=0.0, eqfeas=NaN, obj=NaN, tol=NaN)
        end
        try
            nvar, ncon = nlp.meta.nvar, nlp.meta.ncon
            t = @elapsed r = madnlp(nlp; kkt_system = kkt, max_wall_time = MAXWALL, print_level = MadNLP.ERROR)
            return (name=name, kkt=string(nameof(kkt)), nvar=nvar, ncon=ncon,
                    status=string(r.status), iter=r.iter, nfact=r.counters.factorization_cnt,
                    time=t, eqfeas=_eqfeas(nlp, r.solution), obj=Float64(r.objective), tol=r.options.tol)
        catch e
            return (name=name, kkt=string(nameof(kkt)), nvar=nlp.meta.nvar, ncon=nlp.meta.ncon,
                    status="ERROR:" * _san(first(split(sprint(showerror, e), '\n'))),
                    iter=-1, nfact=-1, time=0.0, eqfeas=NaN, obj=NaN, tol=NaN)
        finally
            finalize(nlp)
        end
    end
end

jobs = [(p, kkt) for kkt in (MadNLP.SparseKKTSystem, MadNLP.SparseCondensedKKTSystem) for p in good]
println("solving $(length(jobs)) (problem × kkt) on $NW workers...")
results = pmap(j -> solve_one(j[1], j[2]), jobs;
               on_error = ex -> (:__EXC, string(typeof(ex)), first(split(sprint(showerror, ex), '\n'))))
let excs = [r for r in results if r isa Tuple && length(r) == 3 && r[1] === :__EXC]
    if !isempty(excs)
        println("!! $(length(excs)) non-NamedTuple results; sample exceptions:")
        for e in unique(excs)[1:min(5, end)]; println("   ", e[2], " | ", e[3]); end
    end
end

# ---- write CSV ----
outdir = joinpath(HERE, "results", "cpu_baseline"); mkpath(outdir)
csv = joinpath(outdir, "$REGIME.csv")
cols = (:name, :kkt, :nvar, :ncon, :status, :iter, :nfact, :time, :eqfeas, :obj, :tol)
open(csv, "w") do io
    println(io, join(cols, ","))
    for p in predecode_failed, kkt in ("SparseKKTSystem", "SparseCondensedKKTSystem")
        println(io, join((p, kkt, -1, -1, "DECODE_ERROR", -1, -1, 0.0, NaN, NaN, NaN), ","))
    end
    for (j, res) in zip(jobs, results)
        st = res isa Tuple && length(res) == 3 && res[1] === :__EXC ? _san("CRASH:" * res[3]) : "CRASH"
        row = res isa NamedTuple ? res :
              (name=j[1], kkt=string(nameof(j[2])), nvar=-1, ncon=-1, status=st,
               iter=-1, nfact=-1, time=0.0, eqfeas=NaN, obj=NaN, tol=NaN)
        println(io, join((row[c] for c in cols), ","))
    end
end
println("wrote $csv")

for kkt in ("SparseKKTSystem", "SparseCondensedKKTSystem")
    rows = [r for r in results if r isa NamedTuple && r.kkt == kkt]
    succ = count(r -> r.status == "SOLVE_SUCCEEDED", rows)
    acc  = count(r -> occursin("ACCEPTABLE", r.status), rows)
    println("$kkt: $(length(rows)) solves  SUCCEEDED=$succ  ACCEPTABLE=$acc  OTHER=$(length(rows)-succ-acc)")
end
println("REGIME_CPU_DONE")
