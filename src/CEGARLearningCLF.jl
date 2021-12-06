module CEGARLearningCLF

using LinearAlgebra
using JuMP
using MathOptInterface
using Printf

abstract type Template{D} end
abstract type Polyhedral{D} <: Template{D} end
struct LearnPolyhedralPoints{D} <: Polyhedral{D} end
struct VerifyPolyhedralSingle{D} <: Polyhedral{D} end
struct VerifyPolyhedralMultiple{D} <: Polyhedral{D} end
struct CEGARProblem{D,LT<:Template{D},VT<:Template{D}}
    A_list::Vector{Matrix{Float64}}
end

state_dim(::CEGARProblem{D}) where D = D
# learn_meth(::CEGARProblem{D,LT,VT}) where {D,LT,VT} = LT
# verify_meth(::CEGARProblem{D,LT,VT}) where {D,LT,VT} = VT
state_dim(::Template{D}) where D = D

include("learner.jl")
include("verifier.jl")
include("process.jl")

end # module
