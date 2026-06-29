#=
    NCL helper(s). Ported from the standalone MadNCL package (`madncl_reference/src/utils.jl`).
    Only the CPU symmetric mat-vec is needed here; the GPU `@kernel` helpers in the reference
    live in MadNLP's GPU extension, not in this CPU path.
=#

function symul!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector, alpha::Number, beta::Number)
    return mul!(y, Symmetric(A, :L), x, alpha, beta)
end
