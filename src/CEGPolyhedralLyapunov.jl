module CEGPolyhedralLyapunov

using LinearAlgebra
using JuMP
using MathOptInterface
using Printf

const _VT_ = Vector{Float64}
const _AT_ = Matrix{Float64}

struct Flow
    point::_VT_
    grads::Vector{_VT_}
end

function Flow(point::AbstractVector, grads::Vector{<:AbstractVector})
    return Flow(_VT_(point), _VT_.(grads))
end

struct Witness
    flow::Flow
    index::Int
end

struct Node
    witness::Witness
    index::Int
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
function make_flows(systems, points)
    flows = Flow[]
    for x in points
        for sys in systems
            any(sys.domain*x .> 0) && continue
            push!(flows, Flow(x, map(A -> A*x, sys.fields)))
        end
    end
    return flows
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

## Includes
include("collection.jl")

include("learner.jl")
include("verifier.jl")
include("process.jl")

end # module
