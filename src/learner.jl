## Learner

struct Witness
    loc::Int
    point::Point
end

struct Learner
    nvar::Int
    nloc::Int
    sys::System
    τ::Float64
    ϵ::Float64
    δ::Float64
    witnesses::Vector{Witness}
    tols::Dict{Symbol,Float64}
end

function Learner(
        nvar::Int, nloc::Int, sys::System, τ::Float64, ϵ::Float64, δ::Float64
    )
    tols = Dict([
        :rad => eps(1.0),
        :pos => -eps(1.0),
        :liedisc => -eps(1.0),
        :liecont => -eps(1.0),
        :norm => eps(1.0)
    ])
    return Learner(nvar, nloc, sys, τ, ϵ, δ, Witness[], tols)
end

set_tol!(lear::Learner, s::Symbol, tol::Float64) = (lear.tols[s] = tol)

function add_witness!(lear::Learner, loc::Int, point::Point)
    npoint = norm(point, Inf)
    if npoint < lear.tols[:norm]*lear.nvar
        error(string("Point norm close to zero: ", npoint))
    end
    @assert 1 ≤ loc ≤ lear.nloc
    push!(lear.witnesses, Witness(loc, point/npoint))
end

## Learn Lyapunov

function _add_evidences!(gen, sys, τ, wit)
    point1 = wit.point
    loc1 = wit.loc
    i1 = add_lf!(gen, loc1)
    npoint1 = norm(point1, Inf) # new
    # npoint = norm(point) # old
    add_evidence_pos!(gen, loc1, i1, point1, npoint1)
    for piece in sys.disc_pieces
        !(loc1 == piece.loc1 && point1 ∈ piece.domain) && continue
        A = piece.A
        point2 = A*point1
        loc2 = piece.loc2
        npoint2 = norm(point2, Inf) # new
        # nderiv = norm(deriv) # old
        ndiff = norm(point2 - point1, Inf)
        nA = opnorm(A, Inf)
        nD = opnorm(A - I, Inf)
        add_evidence_lie!(
            gen, loc1, i1, point1, loc2, point2,
            npoint1, npoint2, ndiff, nA, nD
        )
    end
    for piece in sys.cont_pieces
        !(loc1 == piece.loc && point1 ∈ piece.domain) && continue
        A = piece.A
        diff = τ*(A*point1)
        point2 = point1 + diff
        loc2 = piece.loc
        npoint2 = norm(point2, Inf) # new
        # nderiv = norm(deriv) # old
        ndiff = norm(diff, Inf)
        nA = opnorm(I + τ*A, Inf)
        nD = opnorm(A, Inf)*τ
        add_evidence_lie!(
            gen, loc1, i1, point1, loc2, point2,
            npoint1, npoint2, ndiff, nA, nD
        )
    end
end

function _add_predicates!(verif, nvar, sys)
    for piece in sys.disc_pieces
        add_predicate_pos!(verif, nvar, piece.domain, piece.loc1)
        add_predicate_lie_disc!(
            verif, nvar, piece.domain, piece.loc1, piece.A, piece.loc2
        )
    end
    for piece in sys.cont_pieces
        add_predicate_pos!(verif, nvar, piece.domain, piece.loc)
        add_predicate_lie_cont!(
            verif, nvar, piece.domain, piece.loc, piece.A
        )
    end
end

function _verify_with_exit(
        verif, mpf, tol_pos, tol_liedisc, tol_liecont, solver, do_print
    )
    # new Eccentricity V1:
    do_print && print("|--- Verify pos... ")
    x, r_pos, loc = verify_pos(verif, mpf, solver)
    if r_pos > tol_pos
        do_print && println("CE found: ", x, ", ", r_pos, ", ", loc)
        return x, loc, r_pos, -Inf, -Inf
    else
        do_print && println("No CE found: ", r_pos)
    end # end new Eccentricity V1
    # r_pos = Inf # new Eccentricity V2
    do_print && print("|--- Verify lie disc... ")
    x, r_liedisc, loc = verify_lie_disc(verif, mpf, solver)
    if r_liedisc > tol_liedisc
        do_print && println("CE found: ", x, ", ", r_liedisc, ", ", loc)
        return x, loc, r_pos, r_liedisc, -Inf
    else
        do_print && println("No CE found: ", r_liedisc)
    end
    do_print && print("|--- Verify lie cont... ")
    x, r_liecont, loc = verify_lie_cont(verif, mpf, solver)
    if r_liecont > tol_liecont
        do_print && println("CE found: ", x, ", ", r_liedisc, ", ", loc)
        return x, loc, r_pos, r_liedisc, r_liecont
    else
        do_print && println("No CE found: ", r_liecont)
    end
    return Float64[], 0, r_pos, r_liedisc, r_liecont
end

@enum StatusCode begin
    NOT_SOLVED = 0
    LYAPUNOV_FOUND = 1
    LYAPUNOV_INFEASIBLE = 2
    RADIUS_TOO_SMALL = 3
    MAX_ITER_REACHED = 4
end

struct EvidenceMap
    pos::BitSet
    liedisc::BitSet
    liecont::BitSet
end

EvidenceMap() = EvidenceMap(BitSet(), BitSet(), BitSet())

mutable struct LearnerSolution
    status::StatusCode
    niter::Int
    mpf_list::Vector{MultiPolyFunc}
    pos_evids::Vector{PosEvidence}
    lied_evids::Vector{LieEvidence}
    allwitmap_list::Vector{EvidenceMap}
    newwitmap_list::Vector{EvidenceMap}
    r_list::Vector{Float64}
    r_pos_list::Vector{Float64}
    r_liedisc_list::Vector{Float64}
    r_liecont_list::Vector{Float64}
end

LearnerSolution() = LearnerSolution(
    NOT_SOLVED, 0, MultiPolyFunc[],
    PosEvidence[], LieEvidence[],
    EvidenceMap[], EvidenceMap[],
    Float64[], Float64[], Float64[], Float64[]
)

function learn_lyapunov!(lear::Learner, iter_max, solver; do_print=true)
    gen = Generator(lear.nvar, lear.nloc)
    sol = LearnerSolution()

    for wit in lear.witnesses
        _add_evidences!(gen, lear.sys, lear.τ, wit)
    end

    verif = Verifier()
    _add_predicates!(verif, lear.nvar, lear.sys)

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
        slack = compute_mpf_feasibility(gen, lear.ϵ, lear.δ, solver)[2]
        if slack < 0
            println(string(
                "System does not admit a Lyapunov function with parameters: ",
                "ϵ: ", lear.ϵ, ", δ: ", lear.δ, ", slack ", slack
            ))
            sol.status = LYAPUNOV_INFEASIBLE
            return sol
        end # end Feasibility check

        # mpf, r = compute_mpf_chebyshev(gen, 1/lear.θ, solver)
        mpf, r = compute_mpf_evidence(gen, solver) # test
        if do_print
            println("|--- radius: ", r)
        end
        push!(sol.mpf_list, mpf)
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

        x, loc, r_pos, r_liedisc, r_liecont = _verify_with_exit(
            verif, mpf,
            lear.tols[:pos], lear.tols[:liedisc], lear.tols[:liecont],
            solver, do_print
        )
        push!(sol.r_pos_list, r_pos)
        push!(sol.r_liedisc_list, r_liedisc)
        push!(sol.r_liecont_list, r_liecont)
        if isempty(x)
            println("No CE found")
            println("Valid CLF: terminated")
            sol.status = LYAPUNOV_FOUND
            return sol
        end

        point = x/norm(x, Inf)
        wit = Witness(loc, point)
        _add_evidences!(gen, lear.sys, lear.τ, wit)
        # push!(sol.counterexample_list, wit)
        # push!(witnesses, wit)
        # push!(sol.witnesses_list, copy(witnesses))
    end
end