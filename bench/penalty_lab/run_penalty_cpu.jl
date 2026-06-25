# Kernel-penalty GRID sweep over a CUTEst regime — CONDENSED system, tol 1e-8 (baseline-matched),
# Cosh kernel + StaticContinuation μ_P schedule. Compares against the MadNLP baseline
# (results/cpu_baseline/<regime>_cond_tol1.0e-8.csv) row-for-row.
#
# Only the equality treatment is swapped (RelaxEquality → KernelPenaltyEquality); kkt_system
# (SparseCondensedKKTSystem), tol (1e-8), linear_solver (MumpsSolver), nlp_scaling (true),
# max_iter (3000) match the default-options baseline. Per-problem wall-time cap = safeguard.
#
# Loops over a CONFIGS grid, reusing ONE worker pool + ONE pre-decode (cache) across all configs.
# Same parallel pattern as run_regime_cpu.jl: serial pre-decode, then parallel decode=false solve.
#
# Usage: julia --project=bench/penalty bench/penalty_lab/run_penalty_cpu.jl [regime] [nworkers]
# Writes results/penalty/<regime>__<config_name>.csv  (one file per config).

using Distributed

const HERE   = @__DIR__
const PROJ   = abspath(joinpath(HERE, "..", "penalty"))
const REGIME = length(ARGS) >= 1 ? ARGS[1] : "le500_eq"
const NW     = min(length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 22, 23)   # shared node: ≤24 cores

# ============ PENALTY GRID (edit me) ============
# Schedule: bump μ_P when get_inf_barrier ≤ c_opt·μ_P^p_opt  AND  ‖s_E‖∞ ≤ s_thresh/μ_P,
#           via  μ_P ← min(muP_max, max(kappa_P·μ_P, μ_P^theta_P)).
# Defaults: ε_P = 1/μ_P (c_opt=1,p_opt=-1), slack gate 1/μ_P (s_thresh=1), muP0=10, muP_max=1e7.
const BASE = (c_opt = 1.0, p_opt = -1.0, s_thresh = 1.0, muP0 = 10.0, muP_max = 1.0e7, tol = 1.0e-8)
const CONFIGS = [
    # purely LINEAR (theta_P = 1 ⇒ bump = kappa_P·μ_P), sweep kappa_P
    (name = "lin_k2",   cfg = merge(BASE, (kappa_P = 2.0,  theta_P = 1.0))),
    (name = "lin_k5",   cfg = merge(BASE, (kappa_P = 5.0,  theta_P = 1.0))),
    (name = "lin_k10",  cfg = merge(BASE, (kappa_P = 10.0, theta_P = 1.0))),
    (name = "lin_k20",  cfg = merge(BASE, (kappa_P = 20.0, theta_P = 1.0))),
    # purely SUPERLINEAR (kappa_P = 1 ⇒ bump = μ_P^theta_P), sweep theta_P
    (name = "sup_t1.1",  cfg = merge(BASE, (kappa_P = 1.0, theta_P = 1.1))),
    (name = "sup_t1.25", cfg = merge(BASE, (kappa_P = 1.0, theta_P = 1.25))),
    (name = "sup_t1.5",  cfg = merge(BASE, (kappa_P = 1.0, theta_P = 1.5))),
    (name = "sup_t1.75", cfg = merge(BASE, (kappa_P = 1.0, theta_P = 1.75))),
    (name = "sup_t2",    cfg = merge(BASE, (kappa_P = 1.0, theta_P = 2.0))),
]
# ================================================

probs = String.(filter(!isempty, strip.(readlines(joinpath(HERE, "regimes", "$REGIME.txt")))))
println("regime=$REGIME problems=$(length(probs)) workers=$NW  configs=$(length(CONFIGS))")

# ---- Phase 1: serial pre-decode (warm the shared cache) ----
using MadNLP, CUTEst, NLPModels
const MASTSIF_PATH = ENV["MASTSIF"]
isbuilt(name) = isfile(joinpath(CUTEst.libsif_path, "lib$(name)_double.so"))
println("pre-decoding $(count(!isbuilt, probs)) missing of $(length(probs))...")
predecode_failed = String[]
for p in probs
    isbuilt(p) && continue
    try; nlp = CUTEstModel(p); finalize(nlp); catch; push!(predecode_failed, p); end
end
good = setdiff(probs, predecode_failed)
println("pre-decode done; failed=$(length(predecode_failed)); solving $(length(good)) × $(length(CONFIGS)) configs")

# ---- Phase 2: parallel penalty solve, looped over the grid ----
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
    function solve_one(name::AbstractString, cfg)
        local nlp
        try
            nlp = CUTEstModel(name; decode = false)
        catch e
            return (name=name, nvar=-1, ncon=-1, status="LOAD_ERROR:" * _san(first(split(sprint(showerror, e), '\n'))),
                    iter=-1, nfact=-1, time=0.0, eqfeas=NaN, obj=NaN, muP0=cfg.muP0, muP_final=NaN, n_bumps=-1)
        end
        sched = StaticContinuation(kappa_P=cfg.kappa_P, theta_P=cfg.theta_P, opt_tol_coef=cfg.c_opt,
                                   opt_tol_exp=cfg.p_opt, s_thresh=cfg.s_thresh, muP_max=cfg.muP_max)
        treat = KernelPenaltyEquality(CoshKernel(); schedule=sched, muP=cfg.muP0)
        try
            nvar, ncon = nlp.meta.nvar, nlp.meta.ncon
            t = @elapsed r = madnlp(nlp; kkt_system=MadNLP.SparseCondensedKKTSystem, tol=cfg.tol,
                                    equality_treatment=treat, max_wall_time=MAXWALL, print_level=MadNLP.ERROR)
            return (name=name, nvar=nvar, ncon=ncon, status=string(r.status), iter=r.iter,
                    nfact=r.counters.factorization_cnt, time=t, eqfeas=_eqfeas(nlp, r.solution),
                    obj=Float64(r.objective), muP0=cfg.muP0, muP_final=sched.muP_cur[], n_bumps=sched.n_bumps[])
        catch e
            return (name=name, nvar=nlp.meta.nvar, ncon=nlp.meta.ncon,
                    status="ERROR:" * _san(first(split(sprint(showerror, e), '\n'))),
                    iter=-1, nfact=-1, time=0.0, eqfeas=NaN, obj=NaN, muP0=cfg.muP0, muP_final=sched.muP_cur[], n_bumps=sched.n_bumps[])
        finally
            finalize(nlp)
        end
    end
end

outdir = joinpath(HERE, "results", "penalty"); mkpath(outdir)
cols = (:name, :nvar, :ncon, :status, :iter, :nfact, :time, :eqfeas, :obj, :muP0, :muP_final, :n_bumps)
summary = Tuple{String,Int,Int,Int}[]
for c in CONFIGS
    println("\n=== config $(c.name)  $(c.cfg) ===")
    results = pmap(p -> solve_one(p, c.cfg), good;
                   on_error = ex -> (:__EXC, string(typeof(ex)), first(split(sprint(showerror, ex), '\n'))))
    csv = joinpath(outdir, "$(REGIME)__$(c.name).csv")
    open(csv, "w") do io
        println(io, join(cols, ","))
        for p in predecode_failed
            println(io, join((p, -1, -1, "DECODE_ERROR", -1, -1, 0.0, NaN, NaN, c.cfg.muP0, NaN, -1), ","))
        end
        for (p, res) in zip(good, results)
            row = res isa NamedTuple ? res :
                  (name=p, nvar=-1, ncon=-1, status=(res isa Tuple ? _san("CRASH:" * res[3]) : "CRASH"),
                   iter=-1, nfact=-1, time=0.0, eqfeas=NaN, obj=NaN, muP0=c.cfg.muP0, muP_final=NaN, n_bumps=-1)
            println(io, join((row[col] for col in cols), ","))
        end
    end
    rows = [r for r in results if r isa NamedTuple]
    succ = count(r -> r.status == "SOLVE_SUCCEEDED", rows)
    acc  = count(r -> occursin("ACCEPTABLE", r.status), rows)
    push!(summary, (c.name, succ, acc, length(rows) - succ - acc))
    println("  $(c.name): SUCCEEDED=$succ ACCEPTABLE=$acc OTHER=$(length(rows)-succ-acc)  → $(basename(csv))")
end

println("\n=== GRID SUMMARY ($REGIME, condensed, cosh, tol=1e-8) ===")
println(rpad("config", 12), lpad("SUCC", 6), lpad("ACC", 5), lpad("OTHER", 7))
for (nm, s, a, o) in summary
    println(rpad(nm, 12), lpad(s, 6), lpad(a, 5), lpad(o, 7))
end
println("PENALTY_GRID_DONE")
