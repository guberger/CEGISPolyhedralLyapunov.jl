module CEGARLearningCLF

using LinearAlgebra
using JuMP
using MathOptInterface
using Printf

abstract type LearnMethod{D} end
struct LearnPolyhedralPoints{D} <: LearnMethod{D} end
struct LearnPolyhedralFixed{D} <: LearnMethod{D}
    n_piece::Int
end
abstract type VerifyMethod{D} end
struct VerifyPolyhedralMultiple{D} <: VerifyMethod{D} end
struct CEGARProblem{D,LT,VT}
    A_list::Vector{Matrix{Float64}}
    meth_learn::LT
    meth_verify::VT
end

struct PiecewiseLinearSystem
    n_mode::Int
    Hs_list::Vector{Vector{Vector{Float64}}}
    As_list::Vector{Vector{Matrix{Float64}}}
end

state_dim(::CEGARProblem{D}) where D = D
# learn_meth(::CEGARProblem{D,LT,VT}) where {D,LT,VT} = LT
# verify_meth(::CEGARProblem{D,LT,VT}) where {D,LT,VT} = VT
state_dim(::LearnMethod{D}) where D = D
state_dim(::VerifyMethod{D}) where D = D

function CEGARProblem{D}(A_list, meth_learn, meth_verify) where D
    LT = typeof(meth_learn)
    VT = typeof(meth_verify)
    return CEGARProblem{D,LT,VT}(A_list, meth_learn, meth_verify)
end

function PiecewiseLinearSystem(A_set::Vector{Matrix{Float64}})
    return PiecewiseLinearSystem(1, [Vector{Float64}[]], [A_set])
end

function PiecewiseLinearSystem(Hs_list, As_list)
    @assert length(Hs_list) == length(As_list)
    return PiecewiseLinearSystem(length(As_list), Hs_list, As_list)
end

include("learner.jl")
include("verifier.jl")
include("process.jl")

end # module
