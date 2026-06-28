# Per-iteration monitor of a single penalty solve: tracks μ_B, μ_P, get_inf_barrier, ‖s_E‖∞, the
# μ_B-update gate (inf_barrier ≤ barrier_tol_factor·μ_B) and the μ_P-bump gate
# (max(inf_barrier,‖s_E‖∞) ≤ 1/μ_P for the defaults), and the filter length (resets show as drops).
# Usage: julia --project=bench/penalty bench/penalty_lab/monitor_solve.jl [PROB1 PROB2 ...]
using MadNLP, CUTEst, LinearAlgebra, Printf

mutable struct Mon <: MadNLP.AbstractUserCallback
    rows::Vector{NamedTuple}
end
function (m::Mon)(solver, mode)
    eh = MadNLP.get_cb(solver).equality_handler
    μB = Float64(MadNLP.get_mu(solver)); μP = Float64(eh.muP[])
    sE = isempty(eh.ind_eqslack) ? 0.0 :
         Float64(norm(view(MadNLP.slack(MadNLP.get_x(solver)), eh.ind_eqslack), Inf))
    push!(m.rows, (k=MadNLP.get_cnt(solver).k, muB=μB, muP=μP,
                   inf_barr=Float64(MadNLP.get_inf_barrier(solver)), sE=sE,
                   filt=length(MadNLP.get_filter(solver))))
    return true
end

probs = isempty(ARGS) ? String.(strip.(readlines(joinpath(@__DIR__,"regimes","le500_eq.txt"))))[1:3] : ARGS
for name in probs
    nlp = CUTEstModel(name; decode=false)
    mon = Mon(NamedTuple[])
    sched = StaticContinuation(kappa_P=2.0, theta_P=1.0)        # lin_k2 (purely linear), muP0=1
    treat = KernelPenaltyEquality(CoshKernel(); schedule=sched, muP=1.0)
    r = madnlp(nlp; kkt_system=MadNLP.SparseCondensedKKTSystem, tol=1e-8, equality_treatment=treat,
               intermediate_callback=mon, max_iter=80, print_level=MadNLP.ERROR)
    @printf("\n### %s  status=%s  iters=%d  n_bumps=%d  muP_final=%.3g\n",
            name, r.status, r.iter, sched.n_bumps[], sched.muP_cur[])
    println("   k     muB       muP      inf_barr     sE     |  10*muB(Bgate)  1/muP(Pgate)  filt  flags")
    prevB = NaN; prevP = NaN
    for row in mon.rows[1:min(end,50)]
        bgate = row.inf_barr <= 10*row.muB
        pgate = max(row.inf_barr, row.sE) <= 1/row.muP
        flags = string(row.muB != prevB ? "muB↓ " : "", row.muP != prevP ? "muP↑ " : "",
                       bgate ? "Bok " : "", pgate ? "Pok" : "")
        @printf("  %3d  %.2e  %.2e  %.2e  %.2e |  %.2e     %.2e    %3d   %s\n",
                row.k, row.muB, row.muP, row.inf_barr, row.sE, 10*row.muB, 1/row.muP, row.filt, flags)
        prevB = row.muB; prevP = row.muP
    end
    finalize(nlp)
end
println("\nMONITOR_DONE")
