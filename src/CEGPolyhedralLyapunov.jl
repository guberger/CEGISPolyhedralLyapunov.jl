module CEGPolyhedralLyapunov

using LinearAlgebra
using JuMP
using MathOptInterface
using Printf

const _VT_ = Vector{Float64}
const _AT_ = Matrix{Float64}

struct Witness
    point::_VT_
    flows::Vector{_VT_}
    index::Int
end

function Witness(point::AbstractVector, flows::Vector{<:AbstractVector}, i::Int)
    return Witness(_VT_(point), _VT_.(flows), i)
end

struct LinearSystem
    domain::_AT_
    fields::Vector{_AT_}
end

function LinearSystem(domain::AbstractMatrix,
                      fields::Vector{<:AbstractMatrix})
    return LinearSystem(_AT_(domain), _AT_.(fields))
end

## Utils

function make_witnesses(systems, points)
    witnesses = Witness[]
    i = 0
    for x in points
        for sys in systems
            fields, H = sys.fields, sys.domain
            any(H*x .> 0) && continue
            i += 1
            push!(witnesses, Witness(x, map(A -> A*x, fields), i))
        end
    end
    return witnesses
end


function make_hypercube(dim::Int)
    x = Vector{_VT_}(undef, 2*dim)
    for i = 1:dim
        x[2*i - 1] = map(j -> j == i ? +1 : 0, 1:dim)
        x[2*i - 0] = map(j -> j == i ? -1 : 0, 1:dim)
    end
    return x
end

get_status(model::Model) = (termination_status(model),
                            primal_status(model),
                            dual_status(model))

include("learner.jl")
include("verifier.jl")
include("process.jl")

end # module
