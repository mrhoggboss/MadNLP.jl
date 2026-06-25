# Regenerate the CUTEst size regimes from the env's CUTEst classf.json (authoritative
# metadata built by CUTEst.jl from each model's NLPModels meta). Writes regimes/*.txt,
# regimes/*_eq.txt, regimes/regimes.json and prints the statistics tables.
#
# Run:  julia --project=bench/penalty bench/penalty_lab/build_regimes.jl
#
# Universe: CUTEst problems with nvar >= ncon (ncon = number of general linear+nonlinear
# constraints; variable bounds are NOT counted), EXCLUDING truly-unconstrained problems
# (ncon==0 AND no variable bounds/fixed vars — never tested). Four nested regimes by nvar cap.
# Buckets (mutually exclusive): bound_only | equality_only | inequality_only | mixed.

using CUTEst, JSON

const CLASSF  = joinpath(pkgdir(CUTEst), "src", "classf.json")
const OUTDIR  = joinpath(@__DIR__, "regimes")
const REGIMES = [("le100", 100.0), ("le500", 500.0), ("le5000", 5000.0), ("full", Inf)]
const BUCKETS = ["bound_only", "equality_only", "inequality_only", "mixed"]  # unconstrained excluded
const OBJTYPES = ["none", "constant", "linear", "quadratic", "sum_of_squares", "other"]
mkpath(OUTDIR)

data = JSON.parsefile(CLASSF)

function meta(p)
    v, c = data[p]["variables"], data[p]["constraints"]
    nvar, ncon = v["number"], c["number"]
    n_eq = c["equality"]
    nbnd = v["bounded_below"] + v["bounded_above"] + v["bounded_both"]
    nfix = v["fixed"]
    return Dict("p" => p, "nvar" => nvar, "ncon" => ncon, "n_eq" => n_eq,
                "n_ineq" => ncon - n_eq, "nbnd" => nbnd, "nfix" => nfix,
                "nlin" => c["linear"], "nnln" => c["nonlinear"],
                "has_bounds" => (nbnd > 0 || nfix > 0),
                "objtype" => data[p]["objtype"], "contype" => data[p]["contype"],
                "regular" => data[p]["regular"])
end

bucket(m) =
    m["ncon"] == 0 ? (m["has_bounds"] ? "bound_only" : "unconstrained") :
    m["n_ineq"] == 0 ? "equality_only" :
    m["n_eq"]  == 0 ? "inequality_only" : "mixed"

allm     = [meta(p) for p in keys(data)]
# universe: nvar >= ncon AND not truly-unconstrained (must have a general constraint OR a bound/fixed var)
universe = sort([m for m in allm if m["nvar"] >= m["ncon"] && (m["ncon"] > 0 || m["has_bounds"])],
                by = m -> (m["nvar"], m["p"]))
n_uncon = count(m -> m["nvar"] >= m["ncon"] && m["ncon"] == 0 && !m["has_bounds"], allm)
println("CUTEst classf.json: $(length(data)) problems; $(length(universe)) with nvar>=ncon ",
        "(excluded $n_uncon truly-unconstrained)\n")

regime_problems(cap) = [m for m in universe if m["nvar"] <= cap]
out = Dict{String,Any}()
summ = Tuple{String,Int,Dict{String,Int},Int}[]   # not `summary` (clashes with Base.summary in soft scope)

for (name, cap) in REGIMES
    R = regime_problems(cap)
    for m in R; m["bucket"] = bucket(m); end
    bc = Dict(b => [m for m in R if m["bucket"] == b] for b in BUCKETS)
    eqb = vcat(bc["equality_only"], bc["mixed"])

    write(joinpath(OUTDIR, "$name.txt"),    join((m["p"] for m in R), "\n") * "\n")
    write(joinpath(OUTDIR, "$(name)_eq.txt"), join((m["p"] for m in eqb), "\n") * "\n")

    println(repeat("=", 78))
    println("REGIME $name  (nvar>=ncon, nvar<=$(cap))   total=$(length(R)) problems")
    println(repeat("-", 78))
    println(rpad("  bucket", 18), lpad("count", 6), "   ", lpad("w/ var-bounds", 13), "  ", lpad("has-nonlin con", 14))
    for b in BUCKETS
        L = bc[b]
        wb = count(m -> m["has_bounds"], L)
        nl = count(m -> m["nnln"] > 0, L)
        println(rpad("  $b", 18), lpad(length(L), 6), "   ", lpad(wb, 13), "  ", lpad(nl, 14))
    end
    println(rpad("  -- EQ-BEARING", 18), lpad(length(eqb), 6), "   (equality_only + mixed = penalty test set)")
    od = Dict(o => count(m -> m["objtype"] == o, R) for o in OBJTYPES)
    println("  objtype: ", join(["$o=$(od[o])" for o in OBJTYPES if od[o] > 0], "  "))
    nv = [m["nvar"] for m in R]; nc = [m["ncon"] for m in R]
    println("  nvar range [$(minimum(nv)),$(maximum(nv))]  ncon range [$(minimum(nc)),$(maximum(nc))]  irregular=$(count(m -> !m["regular"], R))")
    out[name] = Dict("cap" => (isinf(cap) ? nothing : Int(cap)), "problems" => R)
    push!(summ, (name, length(R), Dict(b => length(bc[b]) for b in BUCKETS), length(eqb)))
end

println("\n", repeat("=", 78), "\nCROSS-REGIME SUMMARY (counts)\n", repeat("-", 78))
print(rpad("regime", 8), lpad("total", 7)); for b in BUCKETS; print(lpad(b[1:min(9, end)], 11)); end; println(lpad("eq-bear", 9))
for (name, tot, bc, eqb) in summ
    print(rpad(name, 8), lpad(tot, 7)); for b in BUCKETS; print(lpad(bc[b], 11)); end; println(lpad(eqb, 9))
end

open(joinpath(OUTDIR, "regimes.json"), "w") do io; JSON.print(io, out, 1); end
println("\nWrote regime lists + regimes.json to $OUTDIR")
