# μ_P continuation lab — REPORTING.
# (Responsibility: extract and print the data requested, as neutral numbers/tables. No causal
#  interpretation — these functions report what happened, not why.)

using Printf

# one-line-per-run summary across configs
function summary_table(results)
    @printf("%-28s %-26s %6s %8s %9s %11s %16s\n",
            "config", "status", "iters", "nfact", "time_s", "eqfeas", "obj")
    for r in results
        @printf("%-28s %-26s %6d %8d %9.3f %11.3e %16.8e\n",
                r.name, string(r.status), r.iters, r.n_factor, r.time, r.eqfeas, r.obj)
    end
end

# print selected trajectory columns for one run, thinned to ~`maxrows` lines
function trajectory(res; cols = (:k, :muB, :muP, :inf_pr, :inf_du, :inf_compl, :seq_inf, :n_factor), maxrows = 40)
    rows = res.rec.rows
    isempty(rows) && (println("(no rows for $(res.name))"); return)
    step = max(1, cld(length(rows), maxrows))
    println("# trajectory: $(res.name)  ($(length(rows)) rows, every $step shown)")
    @printf("%s\n", join((rpad(string(c), 12) for c in cols)))
    for i in 1:step:length(rows)
        row = rows[i]
        @printf("%s\n", join((rpad(_fmt(row[c]), 12) for c in cols)))
    end
end
_fmt(x::Integer) = string(x)
_fmt(x::Symbol)  = string(x)
_fmt(x::AbstractFloat) = isnan(x) ? "NaN" : @sprintf("%.4e", x)
_fmt(x) = string(x)

# compare one final-iterate trajectory column across runs
function compare_final(results; col = :seq_inf)
    @printf("%-28s %-16s\n", "config", string(col))
    for r in results
        v = isempty(r.rec.rows) ? NaN : r.rec.rows[end][col]
        @printf("%-28s %-16s\n", r.name, _fmt(v))
    end
end

# full last row of a run (all recorded fields)
function final_row(res)
    isempty(res.rec.rows) && (println("(no rows)"); return)
    row = res.rec.rows[end]
    println("# final row: $(res.name)")
    for c in propertynames(row)
        @printf("  %-10s %s\n", string(c), _fmt(row[c]))
    end
end
