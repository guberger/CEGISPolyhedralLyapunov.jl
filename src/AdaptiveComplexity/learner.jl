using LinearAlgebra
using ..CEGISPolyhedralLyapunov: VerifierPos, VerifierLie, verify
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
    status::StatusCode
    niter::Int
    vecs_list::Vector{Vector{_VT_}}
    rad_list::Vector{Float64}
    val_pos_list::Vector{Float64}
    val_lie_list::Vector{Float64}
    witnesses_list::Vector{Vector{Witness}}
end

LearningProblem(
    nvar::Int, ϵ::Float64, θ::Float64, δ::Float64
) = LearningProblem(
    nvar, System[], ϵ, θ, δ, Float64[], _VT_[],
    eps(1.0), eps(1.0), -eps(1.0), eps(1.0),
    NOT_SOLVED, 0,
    Vector{_VT_}[], Float64[], Float64[], Float64[], Vector{Witness}[]
)

set_tol_rad!(prob::LearningProblem, tol_rad::Float64) = (prob.tol_rad = tol_rad)
set_tol_pos!(prob::LearningProblem, tol_pos::Float64) = (prob.tol_pos = tol_pos)
set_tol_lie!(prob::LearningProblem, tol_lie::Float64) = (prob.tol_lie = tol_lie)

function set_Gs!(prob::LearningProblem, α::Float64)
    G0 = (α - 1)/2
    N = max(0, ceil(Int, -log(prob.θ*G0)/log(α)))
    prob.Gs = [(α^k)*G0 for k = 0:N]
end

function add_system!(prob::LearningProblem, domain::Cone, A::_MT_)
    nA = opnorm(A, Inf)
    if nA < prob.tol_norm*prob.nvar
        error(string("Matrix norm close to zero: ", nA))
    end
    A = A/nA
    push!(prob.systems, System(domain, A))
end

function add_point_init!(prob::LearningProblem, point::_VT_)
    np = norm(point, Inf)
    if np < prob.tol_norm*prob.nvar
        error(string("Point norm close to zero: ", np))
    end
    push!(prob.points_init, point/np)
end

function _add_vecs_point!(vecsgen, systems, point)
    wit = Witness(point)
    for system in systems
        point ∉ system.domain && continue
        deriv = system.A*point
        add_deriv!(wit, deriv)
    end
    i = add_vec!(vecsgen)
    add_witness!(vecsgen, i, wit)
    return wit
end

function _make_verifs(nvar, systems)
    Q = length(systems)
    verifs_pos = Vector{VerifierPos}(undef, Q)
    verifs_lie = Vector{VerifierLie}(undef, Q)
    for (q, system) in enumerate(systems)
        verifs_pos[q] = VerifierPos(nvar, system.domain)
        verifs_lie[q] = VerifierLie(nvar, system.domain, system.A)
    end
    return verifs_pos, verifs_lie
end

function _verify(verifs_pos, verifs_lie, vecs, tol_pos, tol_lie, solver)
    print("Verify pos... ")
    x, val_pos, q = verify(verifs_pos, vecs, solver)
    if val_pos < tol_pos
        println("CE found: ", x, ", ", val_pos, ", ", q)
        return x, val_pos, NaN
    else
        println("No CE found: ", val_pos)
    end
    print("Verify lie... ")
    x, val_lie, q = verify(verifs_lie, vecs, solver)
    if val_lie > tol_lie
        println("CE found: ", x, ", ", val_lie, ", ", q)
        return x, val_pos, val_lie
    else
        println("No CE found: ", val_lie)
    end
    return Float64[], val_pos, val_lie
end

function learn_lyapunov!(prob::LearningProblem, iter_max, solver)
    vecsgen = VecsGenerator(prob.nvar, prob.ϵ, prob.θ, prob.δ, prob.Gs)

    witnesses_init = Witness[]
    for point in prob.points_init
        wit = _add_vecs_point!(vecsgen, prob.systems, point)
        push!(witnesses_init, wit)
    end
    push!(prob.witnesses_list, witnesses_init)

    verifs_pos, verifs_lie = _make_verifs(prob.nvar, prob.systems)

    iter = 0

    while true
        iter += 1
        prob.niter = iter
        if iter > iter_max
            println(string("Max iter exceeded: ", iter))
            prob.status = MAX_ITER_REACHED
            return false
        end

        flag = check_feasibility(vecsgen, solver)
        if !flag
            println(string(
                "System does not admit a Lyapunov function with parameters: ",
                "ϵ: ", prob.ϵ, ", θ: ", prob.θ, ", δ: ", prob.δ
            ))
            prob.status = LYAPUNOV_INFEASIBLE
            return false
        end

        vecs, r = compute_vecs(vecsgen, solver)
        push!(prob.vecs_list, vecs)
        push!(prob.rad_list, r)
        if r < prob.tol_rad
            println(string("Satisfiability radius too small: ", r))
            prob.status = RADIUS_TOO_SMALL
            return false
        end

        x, val_pos, val_lie = _verify(
            verifs_pos, verifs_lie, vecs, prob.tol_pos, prob.tol_lie, solver
        )
        push!(prob.val_pos_list, val_pos)
        push!(prob.val_lie_list, val_lie)
        if isempty(x)
            println("No CE found")
            println("Valid CLF: terminated")
            prob.status = LYAPUNOV_FOUND
            return true
        end

        point = x/norm(x, Inf)
        wit = _add_vecs_point!(vecsgen, prob.systems, point)
        push!(prob.witnesses_list, [wit])
    end
end