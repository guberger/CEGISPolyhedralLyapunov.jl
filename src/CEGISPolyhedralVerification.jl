module CEGISPolyhedralVerification

export LinForm, _eval, Witness, Cone, Polyhedron

using LinearAlgebra
using JuMP

_RSC_ = JuMP.MathOptInterface.ResultStatusCode
_TSC_ = JuMP.MathOptInterface.TerminationStatusCode
_VT_ = Vector{Float64}
_MT_ = Matrix{Float64}

include("polyhedra.jl")

struct LinForm
    lin::_VT_
end
_eval(lf::LinForm, point) = dot(lf.lin, point)

struct PolyFunc
    lfs::Vector{LinForm}
    loc_map::Vector{BitSet}
end

struct Piece
    domain::Cone
    loc1::Int
    A::_MT_
    loc2::Int
end

struct System
    pieces::Vector{Piece}
end

System() = System(Piece[])

function add_piece!(sys::System, piece::Piece)
    push!(sys.pieces, piece)
end

function add_piece!(sys::System, domain, A)
    add_piece!(sys, Piece(domain, A))
end

include("generator.jl")
include("verifier.jl")
include("learner.jl")

end # module
