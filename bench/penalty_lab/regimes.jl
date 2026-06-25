# Loader for the precomputed CUTEst size regimes (see bench/penalty_lab/regimes/).
# Built from CUTEst 1.4.0 classf.json by build_regimes.jl. Universe: problems with
# nvar >= ncon (ncon = general linear+nonlinear constraints, NOT variable bounds),
# EXCLUDING truly-unconstrained problems (ncon==0 and no bounds), split by a nvar cap.
#
# PRIMARY (what we test on) — equality-bearing only (equality_only + mixed):
#     :le100_eq  :le500_eq  :le5000_eq  :full_eq
# Secondary — all constrained problems in the size class (bound_only + inequality_only
# + equality_only + mixed):
#     :le100     :le500     :le5000     :full
# All regimes are nested by size: le100* ⊂ le500* ⊂ le5000* ⊂ full*.

import JSON   # for regime_meta; present in the bench/penalty env (load_regime itself needs no deps)

const REGIME_DIR    = joinpath(@__DIR__, "regimes")
const SIZE_CLASSES  = (:le100, :le500, :le5000, :full)
const EQ_REGIMES    = (:le100_eq, :le500_eq, :le5000_eq, :full_eq)  # primary test set
const REGIME_NAMES  = (SIZE_CLASSES..., EQ_REGIMES...)

_size_class(which::Symbol) = Symbol(replace(string(which), r"_eq$" => ""))
_is_eq(which::Symbol)      = endswith(string(which), "_eq")

"""
    load_regime(which::Symbol) -> Vector{String}

Problem names in a regime. `which` is one of:
  primary  (equality-bearing): :le100_eq, :le500_eq, :le5000_eq, :full_eq
  all-constrained            : :le100,    :le500,    :le5000,    :full
"""
function load_regime(which::Symbol)
    which in REGIME_NAMES || error("regime must be one of $REGIME_NAMES, got :$which")
    f = joinpath(REGIME_DIR, string(which, ".txt"))
    return filter(!isempty, strip.(readlines(f)))
end

"""
    regime_meta(which::Symbol) -> Vector{Dict}

Per-problem metadata (nvar, ncon, n_eq, n_ineq, nbnd, nfix, nlin, nnln, bucket, objtype, …)
for every problem in a regime, from regimes/regimes.json. For an `_eq` regime, only the
equality-bearing problems (bucket equality_only or mixed) are returned.
"""
function regime_meta(which::Symbol)
    which in REGIME_NAMES || error("regime must be one of $REGIME_NAMES, got :$which")
    d = JSON.parsefile(joinpath(REGIME_DIR, "regimes.json"))
    probs = d[string(_size_class(which))]["problems"]
    return _is_eq(which) ? [m for m in probs if m["bucket"] in ("equality_only", "mixed")] : probs
end

"Problem names in a regime restricted to a constraint bucket (e.g. \"mixed\", \"equality_only\")."
regime_bucket(which::Symbol, bucket::AbstractString) =
    String[m["p"] for m in regime_meta(which) if m["bucket"] == bucket]
