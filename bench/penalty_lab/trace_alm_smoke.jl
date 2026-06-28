# ALM smoke: solve a few 62-bucket members (which the pure quad penalty left SUCCEEDED-not-tight)
# with the new AugmentedLagrangian schedule + QuadraticKernel. Each SHOULD now reach eqfeas≤1e-8
# (tight) at FINITE μ_P via λ→y. Plus a regression that the default (non-penalty) path still solves.
# Usage: julia --project=bench/penalty bench/penalty_lab/trace_alm_smoke.jl

using MadNLP, CUTEst, NLPModels, Printf
function _eqfeas(nlp, x)
    m = NLPModels.get_ncon(nlp); m == 0 && return 0.0
    c = similar(x, m); NLPModels.cons!(nlp, x, c)
    lc = NLPModels.get_lcon(nlp); uc = NLPModels.get_ucon(nlp)
    return Float64(maximum(abs.(c .- lc) .* (lc .== uc)))
end

println("=== AugmentedLagrangian + QuadraticKernel on 62-bucket members (expect TIGHT eqfeas≤1e-8) ===")
@printf("%-12s %-26s %6s %8s %10s %8s %8s %s\n","name","status","iters","muP_fin","eqfeas","n_lam","n_bump","tight?")
for name in ["HS7","BT2","HS62","MARATOS","BT7","HS14","HS39","DEGENLPB","EIGMINA","HONG"]
    nlp = try CUTEstModel(name; decode=false) catch; continue end
    sched = MadNLP.AugmentedLagrangian(kappa_P=10.0, theta_P=1.0, eta_lambda=0.25, muP_max=1.0e7)
    treat = KernelPenaltyEquality(QuadraticKernel(); schedule=sched, muP=10.0)
    local r
    try
        r = madnlp(nlp; kkt_system=MadNLP.SparseCondensedKKTSystem, tol=1e-8, acceptable_tol=0.0,
                   equality_treatment=treat, max_iter=3000, max_wall_time=60.0, print_level=MadNLP.ERROR)
    catch e; @printf("%-12s ERROR %s\n", name, first(split(sprint(showerror,e),'\n'))); finalize(nlp); continue end
    ef = _eqfeas(nlp, r.solution); tight = string(r.status)=="SOLVE_SUCCEEDED" && ef<=1e-8
    @printf("%-12s %-26s %6d %8.0f %10.2e %8d %8d %s\n", name, string(r.status), r.iter,
            sched.muP_cur[], ef, sched.n_updates[], sched.n_bumps[], tight ? "✓ TIGHT" : "")
    finalize(nlp)
end

println("\n=== regression: default treatment (no penalty) still solves ===")
for name in ["HS6","HS14"]
    nlp = try CUTEstModel(name; decode=false) catch; continue end
    r = madnlp(nlp; kkt_system=MadNLP.SparseCondensedKKTSystem, tol=1e-8, print_level=MadNLP.ERROR)
    @printf("  %-10s %s  eqfeas=%.1e\n", name, string(r.status), _eqfeas(nlp, r.solution))
    finalize(nlp)
end
println("ALM_SMOKE_DONE")
