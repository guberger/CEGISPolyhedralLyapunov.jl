module CEGPolyhedralLyapunov

using StaticArrays
using LinearAlgebra
using JuMP
using MathOptInterface
using Printf

struct Witness{D}
    point::SVector{D,Float64}
    flows::Vector{SVector{D,Float64}}
end

function Witness(point::SVector{D}, flows::Vector{<:SVector{D}}) where D
    return Witness(Val(D), point, flows)
end

function Witness(::Val{D},
                 point::AbstractVector,
                 flows::Vector{<:AbstractVector}) where D
    return Witness{D}(SVector{D,Float64}(point), SVector{D,Float64}.(flows))
end

state_dim(::Witness{D}) where D = D
state_dim(::Type{<:Witness{D}}) where D = D

struct LinearSystem{D}
    consts::Vector{SVector{D,Float64}}
    fields::Vector{SMatrix{D,D,Float64}}
end

function LinearSystem(consts::Vector{<:SVector{D}},
                      fields::Vector{<:SMatrix{D,D}}) where {D,M}
    return LinearSystem(Val(D), consts, fields)
end

function LinearSystem(::Val{D},
                      consts::Vector{<:AbstractVector},
                      fields::Vector{<:AbstractMatrix}) where {D,M}
    return LinearSystem{D}(SVector{D,Float64}.(consts),
                           SMatrix{D,D,Float64}.(fields))
end

state_dim(::LinearSystem{D}) where D = D
state_dim(::Type{<:LinearSystem{D}}) where D = D

## Utils

function make_witnesses(systems, points)
    D = state_dim(eltype(systems))
    witnesses = Witness{D}[]
    for x in points
        for sys in systems
            fields, consts = sys.fields, sys.consts
            any(h -> dot(h, x) > 0, consts) && continue
            push!(witnesses, Witness(Val(D), x, map(A -> A*x, fields)))
        end
    end
    return witnesses
end


function make_hypercube(::Val{D}) where D
    x = Vector{SVector{D,Float64}}(undef, 2*D)
    for i = 1:D
        x[2*i - 1] = SVector{D,Float64}(ntuple(j -> j == i ? +1 : 0, Val(D)))
        x[2*i - 0] = SVector{D,Float64}(ntuple(j -> j == i ? -1 : 0, Val(D)))
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
