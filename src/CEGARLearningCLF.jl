module CEGARLearningCLF

using LinearAlgebra
using JuMP
using MathOptInterface
using Printf

const MOI_term = MathOptInterface.TerminationStatusCode
const MOI_result = MathOptInterface.ResultStatusCode
const OPT_status_T = Tuple{MOI_term,MOI_result,MOI_result}
const OPT_status = NamedTuple{(:termination,:primal,:dual),OPT_status_T}

get_def(x::NamedTuple, field, default) = haskey(x, field) ? x[field] : default

abstract type Polyhedral end
struct PolyhedralPointwise <: Polyhedral
    dim::Int
end

state_dim(m::PolyhedralPointwise) = m.dim

include("learner.jl")
include("verifier.jl")
include("learning_process.jl")

end # module
