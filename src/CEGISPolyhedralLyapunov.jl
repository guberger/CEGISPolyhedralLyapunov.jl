module CEGISPolyhedralLyapunov

# export LinForm, _eval, Witness, Cone, Polyhedron

using LinearAlgebra
using JuMP

_RSC_ = JuMP.MathOptInterface.ResultStatusCode
_TSC_ = JuMP.MathOptInterface.TerminationStatusCode
_VT_ = Vector{Float64}
_MT_ = Matrix{Float64}
Point = _VT_
Deriv = _VT_

include("polyhedra.jl")

struct LinForm
    lin::_VT_
end
_eval(lf::LinForm, point) = dot(lf.lin, point)

struct PolyFunc
    lfs::Vector{LinForm}
end

PolyFunc() = PolyFunc(LinForm[])

add_lf!(pf::PolyFunc, lin) = push!(pf.lfs, LinForm(lin))

struct MultiPolyFunc
    pfs::Vector{PolyFunc}
end

MultiPolyFunc(nloc::Int) = MultiPolyFunc([PolyFunc() for loc = 1:nloc])

add_lf!(mpf::MultiPolyFunc, loc, lin) = add_lf!(mpf.pfs[loc], lin)

struct PieceDisc
    domain::Cone
    loc1::Int
    A::_MT_
    loc2::Int
end

struct PieceCont
    domain::Cone
    loc::Int
    A::_MT_
end

struct System
    disc_pieces::Vector{PieceDisc}
    cont_pieces::Vector{PieceCont}
end

System() = System(PieceDisc[], PieceCont[])

function add_piece_disc!(sys::System, domain, loc1, A, loc2)
    push!(sys.disc_pieces, PieceDisc(domain, loc1, A, loc2))
end

function add_piece_cont!(sys::System, domain, loc, A)
    push!(sys.cont_pieces, PieceCont(domain, loc, A))
end

include("generator.jl")
include("verifier.jl")
include("learner.jl")

end # module
