# Decode CUTEst models and print actual NLPModels meta — to cross-check classf.json.
# Usage: julia --project=bench/penalty bench/penalty_lab/verify_meta.jl PROB1 PROB2 ...
# Prints one line per problem: NAME nvar=.. ncon=.. n_eq=.. n_ineq=.. nbnd=.. nfix=.. nlin=.. nnln=..
using CUTEst, NLPModels
for name in ARGS
    try
        nlp = CUTEstModel(name)
        m = nlp.meta
        n_eq   = length(m.jfix)
        n_ineq = length(m.jlow) + length(m.jupp) + length(m.jrng)
        nbnd   = length(m.ilow) + length(m.iupp) + length(m.irng)
        nfix   = length(m.ifix)
        println("RESULT $name nvar=$(m.nvar) ncon=$(m.ncon) n_eq=$n_eq n_ineq=$n_ineq ",
                "nbnd=$nbnd nfix=$nfix nlin=$(m.nlin) nnln=$(m.nnln)")
        finalize(nlp)
    catch e
        println("ERROR $name $(sprint(showerror, e))")
    end
end
