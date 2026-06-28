# Implementation plan — general equality **penalty-kernel** treatment in MadNLP

Audience: a coding agent working at the root of this fork (`/home/yifan_xu/MadNLP.jl`).
All file:line refs are into this fork; verify each against the live code before editing (they may
drift by a few lines). Work on a branch (e.g. `penalty-kernel`). Do NOT commit; stop for review at
each stage gate.

## 0. Goal
Add a third equality treatment: each equality `c_E(x) = b` gets a **free slack `s`** (coupling
`c_E(x) − s = b`) and a smooth **penalty `φ(μP, s)`** added to the objective, with `μP → ∞` over the
solve. Must be **kernel-general** (cosh, normalized cosh, quadratic; easy to add more) and must work in
**both** the augmented (`SparseKKTSystem`, MA27/indefinite) and condensed (`SparseCondensedKKTSystem`,
SPD/GPU) KKT systems.

## 1. Design decisions (validated against the code)
- **Penalty kernel = a small abstraction.** `AbstractEqualityKernel` with three methods
  `kernel_val/kernel_grad/kernel_hess(kernel, μP, s)` → `φ`, `∂φ/∂s`, `∂²φ/∂s²(=D_s)`. Concrete:
  `CoshKernel`, `CoshNormalizedKernel`, `QuadraticKernel`. (Quadratic is the convex, overflow-free
  baseline — implement and FD-gate it FIRST.)
- **New `EqualityTreatment` subtype: YES.** Treatments may be field-carrying structs (precedent:
  `MakeParameter{T,VT,VI}` at `Callbacks/nlpmodels.jl:57-65`). Add
  `KernelPenaltyEquality{K,T,VT,VI} <: AbstractEqualityTreatment` carrying the kernel, a mutable `μP`,
  the schedule params, the stashed targets `b`, and the penalty-slack indices.
- **New `KKTSystem` subtypes: NO — do not add them.** The penalty's only matrix footprint is a positive
  diagonal `D_s` on the equality-slack block, which the existing systems already accommodate via
  `pr_diag`:
  - augmented `SparseKKTSystem`: slack diagonal is `pr_diag[n+1:n+m]` (`IPM/kernels.jl set_aug_diagonal!`);
    the Lagrangian-Hessian sparsity is the model's (no slack entries), so `D_s` rides `pr_diag`.
  - condensed `SparseCondensedKKTSystem`: `hess_raw` is `n×n` only (`KKT/Sparse/condensed.jl:103`),
    `build_kkt!` (`:354-366`) builds `diag_buffer = Σs/(1−Σd·Σs)` from `pr_diag[n+1:n+m]` and folds
    `Jtᵀ diag(diag_buffer) Jt` into (1,1). Setting `pr_diag[slack] = D_s` ⇒ condensation `= W + Jᵀ D_s J`
    (SPD) with **no structural change**. The `n_slack != m` guard (`condensed.jl:68`) is **satisfied**
    because we slack every constraint (route equalities to `ind_ineq`).
- **Penalty split (load-bearing):**
  - objective **value + gradient** → add in the eval wrappers (so the **filter** and RHS see the penalty);
  - **Hessian `D_s`** → inject into `pr_diag` (NOT the objective Hessian — the condensed Hessian has no
    slack block). Uniform for both systems.
- **Inertia:** augmented penalty equality = indefinite saddle → MA27 + `inertia_correction!` (identical
  in spirit to the validated wrapper); condensed = SPD → `is_inertia_correct` (`condensed.jl:138-140`:
  `num_zero==0 && num_pos==size`) is met by `D_s>0`, so Cholesky/no correction. Existing inertia logic
  needs **no change** (confirm during Stage 3).

## 2. The kernel abstraction (new file `src/IPM/equality_kernels.jl`, `include` from `MadNLP.jl`)
```julia
abstract type AbstractEqualityKernel end
# all three at (μP, s):  value φ,  slack-gradient ∂φ/∂s,  slack-Hessian ∂²φ/∂s² (= D_s > 0)
kernel_val(::AbstractEqualityKernel, μP, s) end
kernel_grad(::AbstractEqualityKernel, μP, s) end
kernel_hess(::AbstractEqualityKernel, μP, s) end

struct QuadraticKernel <: AbstractEqualityKernel end
kernel_val(::QuadraticKernel, μP, s)  = (μP/2)*s^2
kernel_grad(::QuadraticKernel, μP, s) = μP*s
kernel_hess(::QuadraticKernel, μP, s) = μP                     # constant D_s, convex, no overflow

# overflow-safe cosh/sinh (clamp |μP*s| at 709; copy the saturation logic from ExaNL src/model.jl)
struct CoshKernel <: AbstractEqualityKernel end                # φ = cosh(μP s)
kernel_val(::CoshKernel, μP, s)  = _csh(μP*s)
kernel_grad(::CoshKernel, μP, s) = μP*_snh(μP*s)
kernel_hess(::CoshKernel, μP, s) = μP^2*_csh(μP*s)

struct CoshNormalizedKernel <: AbstractEqualityKernel end       # φ = cosh(μP s)/μP
kernel_val(::CoshNormalizedKernel, μP, s)  = _csh(μP*s)/μP
kernel_grad(::CoshNormalizedKernel, μP, s) = _snh(μP*s)
kernel_hess(::CoshNormalizedKernel, μP, s) = μP*_csh(μP*s)
```
Export the kernels and `KernelPenaltyEquality` from `src/MadNLP.jl`.

## 3. The treatment struct (`src/Callbacks/nlpmodels.jl`, next to `EnforceEquality`/`RelaxEquality` ~83-105)
```julia
struct KernelPenaltyEquality{K<:AbstractEqualityKernel, T, VT, VI} <: AbstractEqualityTreatment
    kernel::K
    μP::Base.RefValue{T}        # current penalty parameter (mutable; updated each iter)
    μP0::T; μB0::T; μP_max::T; ρ::T; sat_cap::T   # schedule (ρ≈0.85; sat_cap caps μP*‖s‖)
    b::VT                       # stashed equality targets, one per equality row
    ind_eqslack::VI             # slack-vector positions that are penalty equalities (the eq rows)
end
```
Note: `equality_treatment` is passed as a *type* in options (`options.jl:147`), then instantiated via
`equality_handler = equality_treatment()` (`nlpmodels.jl:454,529`). Because this treatment needs the
model to size `b`/`ind_eqslack` and to carry options, EITHER (a) make `equality_treatment` accept a
constructed *instance* (preferred — pass `KernelPenaltyEquality(...; kernel, μP0, ρ, ...)` and have
`_parse_indexes`/`create_callback` size `b`/`ind_eqslack` from `lcon`), or (b) keep the type-based path
and fill `b`/`ind_eqslack` inside `_treat_equality_initialize!`. Confirm which is cleaner once you read
`create_callback` (`~412/510`).

## 4. Callback seams (`src/Callbacks/nlpmodels.jl`)
1. **`_parse_indexes` (~369)** — add a branch for `KernelPenaltyEquality`: route every constraint to
   `ind_ineq` (`ind_ineq = 1:m`, `ind_eq = []`) exactly like the `RelaxEquality` `else` branch (`:376-378`)
   so each equality gets a slack and the `−I` coupling. Record `ind_eqslack` = positions of the original
   equality rows (`findall(lcon .== ucon)`).
2. **`_treat_equality_initialize!` (~575)** — new method: (a) stash `b .= lcon[eq rows]` onto the handler;
   (b) free ONLY the equality rows' bounds: `for i in eq rows: l[i] = -Inf; u[i] = Inf` (do NOT free the
   genuine inequalities). This makes those slacks free (no barrier).
3. **`initialize!` rhs fixup (~628-631)** — after `rhs .= (lcon .== ucon) .* lcon` (`:630`, which now
   zeros the freed equality rows), restore the target for penalty equalities: `rhs[eq rows] .= handler.b`.
   This is the decoupling: slack bounds free (no barrier) AND `rhs = b` so the slack `s = c(x) − b` is the
   equality residual. (Add a small penalty-aware branch here.)

## 5. Objective seams (`src/IPM/callbacks.jl` + `_eval_*` in `nlpmodels.jl ~771-861`)
Slacks are MadNLP-internal, so the penalty on them is added by MadNLP, not the NLP. With `s = slack(x)`
and the penalty slacks `ind_eqslack`:
- **`eval_f_wrapper`** (`callbacks.jl ~1-17`): `obj += Σ_{i∈ind_eqslack} kernel_val(kernel, μP[], s[i])`.
- **`eval_grad_f_wrapper!`** (`~19-37`): `grad_slack[i] += kernel_grad(kernel, μP[], s[i])` for
  `i∈ind_eqslack` (the slack part of the gradient vector; `slack(...)` accessor).
- **Do NOT** touch `eval_lag_hess_wrapper!` — the Hessian goes in via `pr_diag` (Step 6), not here.

## 6. The `D_s` injection — the crux (`src/IPM/kernels.jl set_aug_diagonal!` + `set_aug_rhs!`)
Add a penalty-aware step that runs **after** the standard diagonal/RHS assembly, using the **current** `μP`:
- diagonal: for `i ∈ ind_eqslack`, `pr_diag[n+i] += kernel_hess(kernel, μP[], s[i])`  (= `D_s`).
- RHS: the slack stationarity `∂φ/∂s` is already in the gradient via Step 5, so `set_aug_rhs!` needs no
  extra term *if* the gradient is current; otherwise add `kernel_grad` to the slack RHS entries directly.
Two clean options for where to call it:
  (a) a new function `add_penalty_diagonal!(kkt, solver)` called from `regular!` **right after
      `set_aug_diagonal!`** (`solver.jl:259`); `pr_diag` is a public field (a view into `aug_raw` for the
      augmented system; the source of `diag_buffer` for the condensed), so `build_kkt!` (run later inside
      `inertia_correction!→factorize_wrapper!`, `factorization.jl:23`) picks it up for BOTH systems.
  (b) dispatch inside `set_aug_diagonal!` on the handler type.
Prefer (a): zero edits to the existing `set_aug_diagonal!`/`build_kkt!` bodies; one new line in `regular!`.

## 7. The μP schedule (`src/IPM/solver.jl regular!`)
- μP lives in `handler.μP` (a `Ref`), reached via `get_cb(solver).equality_handler`.
- Update it once per iteration **right after the μB update** `update_barrier!(...)` (`solver.jl:255`) and
  **before** `set_aug_diagonal!` (`:259`), via a helper `update_penalty_mu!(solver)` that no-ops unless the
  handler is a `KernelPenaltyEquality`:
  ```julia
  μB = get_mu(solver)                                  # the just-updated μB
  h.μP[] = min(h.μP_max, h.μP0 * (h.μB0 / μB)^h.ρ)     # static closed form, ρ≈0.85
  h.μP[] = min(h.μP[], h.sat_cap / max(norm(s,Inf), eps()))  # cliff guard
  ```
  (The intermediate_callback at `:239` fires BEFORE the barrier update, so it can't be used here.)
- One-iteration lag: the objective gradient (φ′) was evaluated at the previous iterate's μP, while `D_s`
  (Step 6) uses the new μP. Acceptable for v1 (μP changes slowly); note it. To kill the lag later, also
  refresh the slack RHS with `kernel_grad(new μP)` in the Step-6 step.

## 8. Options (`src/IPM/options.jl`)
Add to `MadNLPOptions` (the `@kwdef mutable struct`, near the equality/kkt defaults `~147/213`):
`penalty_kernel::Type = QuadraticKernel`, `muP_init::T`, `muP_max::T`, `muP_rho::T`, `muP_sat_cap::T`.
They are auto-parsed by `set_options!` (`:7-19`). The user selects the treatment with
`equality_treatment = KernelPenaltyEquality` + these options; thread them into the handler constructor.

## 9. Inertia (verify, likely no change)
- condensed: `is_inertia_correct` (`condensed.jl:138-140`) wants SPD; `D_s>0` keeps it SPD ⇒ fine.
- augmented: target `(num_pos=num_variables, num_neg=m, num_zero=0)` (`KKTsystem.jl ~242-244`); the penalty
  slack adds to `num_pos`, the equality coupling to `num_neg` ⇒ target unchanged. Confirm empirically in
  Stage 1 (watch `del_w` not exploding).

## 10. Validation gates (BLOCKING before each sweep; mirror `ExaNL benchmark/barrier/fd_gate.jl`)
- **FD gate**: with the **QuadraticKernel** + fixed μP on a small mixed CUTEst problem (e.g. HS14), check
  vs central differences (<1e-9): (i) `eval_f` includes `(μP/2)Σs²`; (ii) gradient slack entries `= μP·s`;
  (iii) assembled `pr_diag[slack] = μP`; (iv) a full Newton step matches the dense penalized KKT solve.
  Then repeat for `CoshKernel`.
- **No-op/equivalence control**: QuadraticKernel with very large fixed μP should approach EnforceEquality
  on a few problems (tight equality); also reconcile against the ExaNL cosh wrapper results.
- **Smoke**: solve a handful of mixed problems under both `SparseKKTSystem` and `SparseCondensedKKTSystem`.

## 11. Staged build order (stop for review at each ✋)
1. **Augmented + QuadraticKernel + FIXED μP.** Steps 2–6 (kernels, treatment, callback seams, objective
   wrappers, `pr_diag` injection) restricted to `SparseKKTSystem`. ✋ FD gate (Quad) + smoke.
2. **Add the μP schedule** (Step 7) + the cliff guard. ✋ smoke; watch the μP/μB pacing.
3. **Condensed** (`SparseCondensedKKTSystem`): confirm `D_s` rides `diag_buffer` (no code change beyond
   Step 6 already writing `pr_diag`), verify SPD inertia. ✋ FD gate (condensed) + smoke; this is the GPU path.
4. **Cosh + normalized-cosh kernels** + options wiring (Step 8). ✋ FD gate (cosh).
5. **Benchmark** vs `RelaxEquality` on the mixed CUTEst subsets (the decisive comparison — cosh's true
   competitor is relax, on the condensed path).

## 12. Open questions / watch-items
- The rhs/slack-bound decoupling (Step 4.3) is the fiddliest seam — get `b` stashed and `rhs=b` restored.
- Filter constants (`s_phi`,`s_theta`,`eta_phi`) are tuned for a log-barrier φ; the cosh penalty dominates
  the merit — may need a look (quadratic is gentler; start there).
- Free penalty slacks must be excluded from the bound-barrier machinery (`ind_lb`/`ind_ub`, fraction-to-
  boundary, `kappa_sigma` reset) — freeing their bounds (Step 4.2) should already exclude them; verify.
- Keep the genuine inequalities barrier-handled (only the *equality* slacks are penalized) — `ind_eqslack`
  is the gate everywhere.
