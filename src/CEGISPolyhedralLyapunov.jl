module CEGISPolyhedralLyapunov

using LinearAlgebra
using JuMP

struct Piece{AT<:AbstractMatrix,VLT<:AbstractVector{<:AbstractVector}}
    A::AT
    lfs_dom::VLT
end

include("generator.jl")
include("verifier.jl")
include("learner.jl")

end # module