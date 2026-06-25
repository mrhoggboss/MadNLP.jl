# Loader for the precomputed CUTEst size regimes (see bench/penalty_lab/regimes/).
# Regimes are the CUTEst problems with nvar >= ncon (ncon = general linear+nonlinear
# constraints, NOT variable bounds), EXCLUDING truly-unconstrained problems (ncon==0 and
# no bounds), split by a nvar cap: :le100, :le500, :le5000, :full (nested).
# `<regime>.txt` = all problems in the regime; `<regime>_eq.txt` = the equality-bearing
# subset (equality_only + mixed) — the set that actually exercises the equality penalty.
# Built from CUTEst 1.4.0 classf.json by build_regimes.jl.

const REGIME_DIR = joinpath(@__DIR__, "regimes")
const REGIME_NAMES = (:le100, :le500, :le5000, :full)

"""
    load_regime(which::Symbol; eq_only::Bool=false) -> Vector{String}

Problem names in a regime. `which ∈ (:le100,:le500,:le5000,:full)`.
`eq_only=true` returns only the equality-bearing problems (equality_only + mixed).
"""
function load_regime(which::Symbol; eq_only::Bool = false)
    which in REGIME_NAMES || error("regime must be one of $REGIME_NAMES, got :$which")
    f = joinpath(REGIME_DIR, string(which, eq_only ? "_eq" : "", ".txt"))
    return filter(!isempty, strip.(readlines(f)))
end

"""
    regime_meta(which::Symbol) -> Vector{Dict}

Per-problem metadata (nvar, ncon, n_eq, n_ineq, nbnd, nfix, nlin, nnln, bucket, objtype, …)
for every problem in a regime, from regimes/regimes.json.
"""
function regime_meta(which::Symbol)
    which in REGIME_NAMES || error("regime must be one of $REGIME_NAMES, got :$which")
    JSON = Base.require(@__MODULE__, :JSON)   # lazy: needs JSON in the active env (bench/penalty)
    d = JSON.parsefile(joinpath(REGIME_DIR, "regimes.json"))
    return d[string(which)]["problems"]
end

"Problem names in a regime restricted to a constraint bucket (e.g. \"mixed\", \"equality_only\")."
function regime_bucket(which::Symbol, bucket::AbstractString)
    return String[m["p"] for m in regime_meta(which) if m["bucket"] == bucket]
end
