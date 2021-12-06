module CEGARLearningCLF

using LinearAlgebra
using JuMP
using MathOptInterface
using Printf

# const MOI_term = MathOptInterface.TerminationStatusCode
# const MOI_result = MathOptInterface.ResultStatusCode
# const OPT_status_T = Tuple{MOI_term,MOI_result,MOI_result}
# const OPT_status = NamedTuple{(:termination,:primal,:dual),OPT_status_T}

get_def(x::NamedTuple, field, default) = haskey(x, field) ? x[field] : default

abstract type Template{D} end
abstract type Polyhedral{D} <: Template{D} end
struct LearnPolyhedralPoints{D} <: Polyhedral{D} end
struct VerifyPolyhedralHRep{D} <: Polyhedral{D} end
struct CEGARProblem{D,LT<:Template{D},VT<:Template{D},AT}
    A_list::AT
end

state_dim(::CEGARProblem{D}) where D = D
state_dim(::Template{D}) where D = D

include("learner.jl")
include("verifier.jl")
# include("learning_process.jl")

end # module
