module CEGPolyhedralLyapunov

using LinearAlgebra
using JuMP
using Printf

const _VT_ = Vector{Float64}
const _AT_ = Matrix{Float64}
const _TSC_ = JuMP.MathOptInterface.TerminationStatusCode
const _RSC_ = JuMP.MathOptInterface.ResultStatusCode

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

abstract type Tree end

mutable struct Root <: Tree end

mutable struct Branch <: Tree
    node::Node
    tail::Tree
end

grow(tree::Tree, node::Node) = Branch(node, tree)

function seed(nodes)
    tree = Root()
    for node in nodes
        tree = grow(tree, node)
    end
    return tree
end

Base.isempty(::Root) = true
Base.isempty(::Branch) = false
Base.iterate(::Tree, ::Root) = nothing
function Base.iterate(tree::Tree, state::Branch=tree)
    state.node, state.tail
end

## Utils
function add_grads!(grads, systems, x)
    for sys in systems
        any(sys.domain*x .> 0) && continue
        for A in sys.fields
            push!(grads, A*x)
        end
    end
    return grads
end

function make_flow(systems, x)
    flow = Flow(x, _VT_[])
    add_grads!(flow.grads, systems, x)
    return flow   
end

function hypercube(dim::Int)
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
include("learner.jl")
include("verifier.jl")
include("process.jl")

end # module
