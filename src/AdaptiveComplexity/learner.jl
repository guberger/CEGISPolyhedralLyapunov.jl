using LinearAlgebra
using ..Verifier: VerifyingProblem, verify_pos, verify_lie
using ..Polyhedra: Cone

_VT_ = Vector{Float64}
_MT_ = Matrix{Float64}

struct System
    domain::Cone
    A::_MT_
end

@enum StatusCode begin
    NOT_SOLVED = 0
    LYAPUNOV_FOUND = 1
    LYAPUNOV_INFEASIBLE = 2
    RADIUS_TOO_SMALL = 3
    MAX_ITER_REACHED = 4
end


mutable struct LearningProblem
    nvar::Int
    systems::Vector{System}
    ϵ::Float64
    θ::Float64
    δ::Float64
    Gs::Vector{Float64}
    points_init::Vector{_VT_}
    tol_rad::Float64
    tol_pos::Float64
    tol_lie::Float64
    tol_norm::Float64
end

LearningProblem(nvar::Int, ϵ::Float64, θ::Float64, δ::Float64) =
    LearningProblem(
        nvar, System[], ϵ, θ, δ, Float64[], _VT_[],
        eps(1.0), eps(1.0), -eps(1.0), eps(1.0)
    )

set_tol_rad!(prob::LearningProblem, tol_rad::Float64) = (prob.tol_rad = tol_rad)
set_tol_pos!(prob::LearningProblem, tol_pos::Float64) = (prob.tol_pos = tol_pos)
set_tol_lie!(prob::LearningProblem, tol_lie::Float64) = (prob.tol_lie = tol_lie)

# function set_Gs!(prob::LearningProblem, α::Float64)
#     G0 = (α - 1)/2
#     N = max(0, ceil(Int, -log(prob.θ*G0)/log(α)))
#     prob.Gs = [(α^k)*G0 for k = 0:N]
# end

function add_G!(prob::LearningProblem, G::Float64)
    push!(prob.Gs, G)
end

function add_system!(prob::LearningProblem, domain::Cone, A::_MT_)
    push!(prob.systems, System(domain, A))
end

function add_point_init!(prob::LearningProblem, point::_VT_)
    np = norm(point, Inf)
    if np < prob.tol_norm*prob.nvar
        error(string("Point norm close to zero: ", np))
    end
    push!(prob.points_init, point/np)
end

function make_witness_from_point(systems, point)
    npoint = norm(point, Inf) # new
    # npoint = norm(point) # old
    pos_constrs = [PosConstraint(point, npoint)]
    lie_constrs = LieConstraint[]
    for system in systems
        point ∉ system.domain && continue
        deriv = system.A*point
        nderiv = norm(deriv, Inf) # new
        # nderiv = norm(deriv) # old
        nA = opnorm(system.A, Inf)
        lie_con = LieConstraint(point, deriv, npoint, nderiv, nA)
        push!(lie_constrs, lie_con)
    end
    return Witness(pos_constrs, lie_constrs)
end

function make_verifs_from_systems(nvar, systems)
    Q = length(systems)
    verifs = Vector{VerifyingProblem}(undef, Q)
    for (q, system) in enumerate(systems)
        verifs[q] = VerifyingProblem(nvar, system.domain, system.A)
    end
    return verifs
end

function _verify(verifs, vecs, tol_pos, tol_lie, solver)
    # new Eccentricity V1:
    print("Verify pos... ")
    x, val_pos, q = verify_pos(verifs, vecs, solver)
    if val_pos < tol_pos
        println("CE found: ", x, ", ", val_pos, ", ", q)
        return x, val_pos, -Inf
    else
        println("No CE found: ", val_pos)
    end # end new Eccentricity V1
    # val_pos = Inf # new Eccentricity V2
    print("Verify lie... ")
    x, val_lie, q = verify_lie(verifs, vecs, solver)
    if val_lie > tol_lie
        println("CE found: ", x, ", ", val_lie, ", ", q)
        return x, val_pos, val_lie
    else
        println("No CE found: ", val_lie)
    end
    return Float64[], val_pos, val_lie
end

mutable struct LearnerSolution
    status::StatusCode
    niter::Int
    witnesses_list::Vector{Vector{Witness}}
    vecs_list::Vector{Vector{_VT_}}
    rad_list::Vector{Float64}
    counterexample_list::Vector{Witness}
    val_pos_list::Vector{Float64}
    val_lie_list::Vector{Float64}
end

LearnerSolution() = LearnerSolution(
    NOT_SOLVED, 0,
    Vector{Witness}[], Vector{_VT_}[], Float64[],
    Witness[], Float64[], Float64[]
)

function learn_lyapunov!(prob::LearningProblem, iter_max, solver)
    vecsgen = VecsGenerator(prob.nvar, prob.Gs)
    sol = LearnerSolution()

    witnesses = Witness[]
    for point in prob.points_init
        wit = make_witness_from_point(prob.systems, point)
        add_witness!(vecsgen, wit)
        push!(witnesses, wit)
    end
    push!(sol.witnesses_list, copy(witnesses))

    verifs = make_verifs_from_systems(prob.nvar, prob.systems)

    iter = 0

    while true
        iter += 1
        print("Iter: ", iter)
        sol.niter = iter
        if iter > iter_max
            println(string("Max iter exceeded: ", iter))
            sol.status = MAX_ITER_REACHED
            return sol
        end

        r = compute_feasibility(vecsgen, prob.ϵ, prob.θ, solver)
        if r < prob.δ
            println(string(
                "System does not admit a Lyapunov function with parameters: ",
                "ϵ: ", prob.ϵ, ", θ: ", prob.θ, ", δ: ", prob.δ, " - ", r
            ))
            sol.status = LYAPUNOV_INFEASIBLE
            return sol
        end

        vecs, r = compute_vecs(vecsgen, solver)
        println(" - radius: ", r)
        push!(sol.vecs_list, vecs)
        push!(sol.rad_list, r)
        if r < prob.tol_rad
            println(string("Satisfiability radius too small: ", r))
            sol.status = RADIUS_TOO_SMALL
            return sol
        end

        # new Eccentricity V2:
        # for k = 1:prob.nvar
        #     vec_side = [(k_ == k ? 1.0 : 0.0) for k_ = 1:prob.nvar]
        #     push!(vecs, vec_side/(2*prob.ϵ))
        #     push!(vecs, -vec_side/(2*prob.ϵ))
        # end # end new Eccentricity V2

        x, val_pos, val_lie = _verify(
            verifs, vecs, prob.tol_pos, prob.tol_lie, solver
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
        wit = make_witness_from_point(prob.systems, point)
        add_witness!(vecsgen, wit)
        push!(sol.counterexample_list, wit)
        push!(witnesses, wit)
        push!(sol.witnesses_list, copy(witnesses))
    end
end