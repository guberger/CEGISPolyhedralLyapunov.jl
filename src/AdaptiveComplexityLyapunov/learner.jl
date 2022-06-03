using LinearAlgebra
using ..Polyhedra: Cone

_VT_ = Vector{Float64}
_MT_ = Matrix{Float64}

struct Piece
    domain::Cone
    A::_MT_
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

## Learner

struct Learner
    nvar::Int
    system::System
    ϵ::Float64
    θ::Float64
    δ::Float64
    points_init::Vector{_VT_}
    tols::Dict{Symbol,Float64}
end

function Learner(nvar::Int, sys::System, ϵ::Float64, θ::Float64, δ::Float64)
    tols = Dict([
        :rad => eps(1.0),
        :pos => eps(1.0),
        :lie => -eps(1.0),
        :norm => eps(1.0)
    ])
    return Learner(nvar, sys, ϵ, θ, δ, _VT_[], tols)
end

set_tol!(lear::Learner, s::Symbol, tol::Float64) = (lear.tols[s] = tol)

function add_point_init!(lear::Learner, point::_VT_)
    np = norm(point, Inf)
    if np < lear.tols[:norm]*lear.nvar
        error(string("Point norm close to zero: ", np))
    end
    push!(lear.points_init, point/np)
end

## Learn Lyapunov

function make_witness_from_point_system(sys, point)
    wit = Witness()
    npoint = norm(point, Inf) # new
    # npoint = norm(point) # old
    add_evidence_pos!(wit, point, npoint)
    for piece in sys.pieces
        point ∉ piece.domain && continue
        deriv = piece.A*point
        nderiv = norm(deriv, Inf) # new
        # nderiv = norm(deriv) # old
        nA = opnorm(piece.A, Inf)
        add_evidence_lie!(wit, point, deriv, npoint, nderiv, nA)
    end
    return wit
end

function make_verif_from_system(nvar, sys)
    verif = Verifier()
    for piece in sys.pieces
        add_verifying_pos!(verif, nvar, piece.domain)
        add_verifying_lie!(verif, nvar, piece.domain, piece.A)
    end
    return verif
end

function _verify_with_exit(verif, vecs, tol_pos, tol_lie, solver, do_print)
    # new Eccentricity V1:
    do_print && print("|--- Verify pos... ")
    x, val_pos, q = verify_pos(verif, vecs, solver)
    if val_pos < tol_pos
        do_print && println("CE found: ", x, ", ", val_pos, ", ", q)
        return x, val_pos, -Inf
    else
        do_print && println("No CE found: ", val_pos)
    end # end new Eccentricity V1
    # val_pos = Inf # new Eccentricity V2
    do_print && print("|--- Verify lie... ")
    x, val_lie, q = verify_lie(verif, vecs, solver)
    if val_lie > tol_lie
        do_print && println("CE found: ", x, ", ", val_lie, ", ", q)
        return x, val_pos, val_lie
    else
        do_print && println("No CE found: ", val_lie)
    end
    return Float64[], val_pos, val_lie
end

@enum StatusCode begin
    NOT_SOLVED = 0
    LYAPUNOV_FOUND = 1
    LYAPUNOV_INFEASIBLE = 2
    RADIUS_TOO_SMALL = 3
    MAX_ITER_REACHED = 4
end

mutable struct LearnerSolution
    status::StatusCode
    niter::Int
    witnesses_list::Vector{Vector{Witness}}
    vecs_list::Vector{Vector{_VT_}}
    r_list::Vector{Float64}
    counterexample_list::Vector{Witness}
    val_pos_list::Vector{Float64}
    val_lie_list::Vector{Float64}
end

LearnerSolution() = LearnerSolution(
    NOT_SOLVED, 0,
    Vector{Witness}[], Vector{_VT_}[], Float64[],
    Witness[], Float64[], Float64[]
)

function learn_lyapunov!(lear::Learner, iter_max, solver; do_print=true)
    gen = Generator(lear.nvar)
    sol = LearnerSolution()

    witnesses = Witness[]
    for point in lear.points_init
        wit = make_witness_from_point_system(lear.system, point)
        add_witness!(gen, wit)
        push!(witnesses, wit)
    end
    push!(sol.witnesses_list, copy(witnesses))

    verif = make_verif_from_system(lear.nvar, lear.system)

    iter = 0

    while true
        iter += 1
        do_print && println("Iter: ", iter)
        sol.niter = iter
        if iter > iter_max
            println(string("Max iter exceeded: ", iter))
            sol.status = MAX_ITER_REACHED
            return sol
        end

        # Feasibility check:
        slack = compute_vecs_feasibility(gen, lear.ϵ, lear.θ, lear.δ, solver)[2]
        if slack < 0
            println(string(
                "System does not admit a Lyapunov function with parameters: ",
                "ϵ: ", lear.ϵ, ", θ: ", lear.θ, ", δ: ", lear.δ,
                ", slack ", slack
            ))
            sol.status = LYAPUNOV_INFEASIBLE
            return sol
        end # end Feasibility check

        # vecs, r = compute_vecs_chebyshev(gen, 1/lear.θ, solver)
        vecs, r = compute_vecs_witness(gen, 1/lear.θ, solver) # test
        if do_print
            println("|--- radius: ", r)
        end
        push!(sol.vecs_list, vecs)
        push!(sol.r_list, r)
        if r < lear.tols[:rad]
            println(string("Satisfiability radius too small: ", r))
            sol.status = RADIUS_TOO_SMALL
            return sol
        end

        # new Eccentricity V2:
        # for k = 1:lear.nvar
        #     vec_side = [(k_ == k ? 1.0 : 0.0) for k_ = 1:lear.nvar]
        #     push!(vecs, vec_side/(2*lear.ϵ))
        #     push!(vecs, -vec_side/(2*lear.ϵ))
        # end # end new Eccentricity V2

        x, val_pos, val_lie = _verify_with_exit(
            verif, vecs, lear.tols[:pos], lear.tols[:lie], solver, do_print
        )
        push!(sol.val_pos_list, val_pos)
        push!(sol.val_lie_list, val_lie)
        if isempty(x)
            println("No CE found")
            println("Valid CLF: terminated")
            sol.status = LYAPUNOV_FOUND
            return sol
        end

        point = x/norm(x, Inf)
        wit = make_witness_from_point_system(lear.system, point)
        add_witness!(gen, wit)
        push!(sol.counterexample_list, wit)
        push!(witnesses, wit)
        push!(sol.witnesses_list, copy(witnesses))
    end
end