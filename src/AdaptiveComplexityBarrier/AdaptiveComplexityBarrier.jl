module AdaptiveComplexityBarrier

using LinearAlgebra
using JuMP
using ..Polyhedra: Cone

_RSC_ = JuMP.MathOptInterface.ResultStatusCode
_TSC_ = JuMP.MathOptInterface.TerminationStatusCode
_VT_ = Vector{Float64}
_MT_ = Matrix{Float64}

struct AffForm
    lin::_VT_
    con::Float64
end

include("generator.jl")
# include("verifier.jl")
# include("learner.jl")

end # module