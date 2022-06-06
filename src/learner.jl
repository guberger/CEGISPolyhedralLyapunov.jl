## Learner

struct Learner
    nvar::Int
    nloc::Int
    system::System
    ϵ::Float64
    δ::Float64
    states_init::Vector{State}
    tols::Dict{Symbol,Float64}
end

function Learner(nvar::Int, nloc::Int, sys::System, ϵ::Float64, δ::Float64)
    tols = Dict([
        :rad => eps(1.0),
        :pos => -eps(1.0),
        :lie => -eps(1.0),
        :norm => eps(1.0)
    ])
    return Learner(nvar, nloc, sys, ϵ, δ, State[], tols)
end

set_tol!(lear::Learner, s::Symbol, tol::Float64) = (lear.tols[s] = tol)

function add_state_init!(lear::Learner, point::_VT_, loc::Int)
    npoint = norm(point, Inf)
    if npoint < lear.tols[:norm]*lear.nvar
        error(string("Point norm close to zero: ", npoint))
    end
    @assert 1 ≤ loc ≤ lear.nloc
    push!(lear.states_init, State(point/npoint, loc))
end

## Learn Lyapunov

function make_witness_from_state_system(sys, state)
    wit = Witness()
    point1 = state.point
    loc1 = state.loc
    state1 = State(point1, loc1)
    npoint1 = norm(point1, Inf) # new
    # npoint = norm(point) # old
    add_evidence_pos!(wit, state1, npoint1)
    for piece in sys.pieces
        !(loc1 == piece.loc1 && point1 ∈ piece.domain) && continue
        point2 = piece.A*point1
        state2 = State(point2, piece.loc2)
        npoint2 = norm(point2, Inf) # new
        # nderiv = norm(deriv) # old
        ndiff = norm(point2 - point1, Inf)
        nA = opnorm(piece.A, Inf)
        nD = opnorm(piece.D, Inf)
        add_evidence_lie!(wit, state1, state2, npoint1, npoint2, ndiff, nA, nD)
    end
    return wit
end

function make_verif_from_system(nvar, sys)
    verif = Verifier()
    for piece in sys.pieces
        add_predicate_pos!(verif, nvar, piece.domain, piece.loc1)
        add_predicate_lie!(
            verif, nvar, piece.domain, piece.loc1, piece.A, piece.loc2
        )
    end
    return verif
end

function _verify_with_exit(verif, polyf, tol_pos, tol_lie, solver, do_print)
    # new Eccentricity V1:
    do_print && print("|--- Verify pos... ")
    x, r_pos, loc = verify_pos(verif, polyf, solver)
    if r_pos > tol_pos
        do_print && println("CE found: ", x, ", ", r_pos, ", ", loc)
        return x, loc, r_pos, -Inf
    else
        do_print && println("No CE found: ", r_pos)
    end # end new Eccentricity V1
    # r_pos = Inf # new Eccentricity V2
    do_print && print("|--- Verify lie... ")
    x, r_lie, loc = verify_lie(verif, polyf, solver)
    if r_lie > tol_lie
        do_print && println("CE found: ", x, ", ", r_lie, ", ", loc)
        return x, loc, r_pos, r_lie
    else
        do_print && println("No CE found: ", r_lie)
    end
    return Float64[], 0, r_pos, r_lie
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
    polyf_list::Vector{PolyFunc}
    r_list::Vector{Float64}
    counterexample_list::Vector{Witness}
    r_pos_list::Vector{Float64}
    r_lie_list::Vector{Float64}
end

LearnerSolution() = LearnerSolution(
    NOT_SOLVED, 0,
    Vector{Witness}[], PolyFunc[], Float64[],
    Witness[], Float64[], Float64[]
)

function learn_lyapunov!(lear::Learner, iter_max, solver; do_print=true)
    gen = Generator(lear.nvar, lear.nloc)
    sol = LearnerSolution()

    witnesses = Witness[]
    for state in lear.states_init
        wit = make_witness_from_state_system(lear.system, state)
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
        slack = compute_polyf_feasibility(gen, lear.ϵ, lear.δ, solver)[2]
        if slack < 0
            println(string(
                "System does not admit a Lyapunov function with parameters: ",
                "ϵ: ", lear.ϵ, ", δ: ", lear.δ, ", slack ", slack
            ))
            sol.status = LYAPUNOV_INFEASIBLE
            return sol
        end # end Feasibility check

        # polyf, r = compute_polyf_chebyshev(gen, 1/lear.θ, solver)
        polyf, r = compute_polyf_witness(gen, solver) # test
        if do_print
            println("|--- radius: ", r)
        end
        push!(sol.polyf_list, polyf)
        push!(sol.r_list, r)
        if r < lear.tols[:rad]
            println(string("Satisfiability radius too small: ", r))
            sol.status = RADIUS_TOO_SMALL
            return sol
        end

        # new Eccentricity V2:
        # for k = 1:lear.nvar
        #     vec_side = [(k_ == k ? 1.0 : 0.0) for k_ = 1:lear.nvar]
        #     push!(lfs, vec_side/(2*lear.ϵ))
        #     push!(lfs, -vec_side/(2*lear.ϵ))
        # end # end new Eccentricity V2

        x, loc, r_pos, r_lie = _verify_with_exit(
            verif, polyf, lear.tols[:pos], lear.tols[:lie], solver, do_print
        )
        push!(sol.r_pos_list, r_pos)
        push!(sol.r_lie_list, r_lie)
        if isempty(x)
            println("No CE found")
            println("Valid CLF: terminated")
            sol.status = LYAPUNOV_FOUND
            return sol
        end

        point = x/norm(x, Inf)
        wit = make_witness_from_state_system(lear.system, State(point, loc))
        add_witness!(gen, wit)
        push!(sol.counterexample_list, wit)
        push!(witnesses, wit)
        push!(sol.witnesses_list, copy(witnesses))
    end
end