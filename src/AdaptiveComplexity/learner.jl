using LinearAlgebra
using ..CEGISPolyhedralLyapunov: VerifierPos, VerifierLie, verify
using ..Polyhedra: Cone

_VT_ = Vector{Float64}
_MT_ = Matrix{Float64}

struct System
    domain::Cone
    A::_MT_
end

mutable struct LearningProblem
    nvar::Int
    systems::Vector{System}
    ϵ::Float64
    θ::Float64
    δ::Float64
    Gs::Vector{Float64}
    points::Vector{_VT_}
    tol_rad::Float64
    tol_pos::Float64
    tol_lie::Float64
    tol_norm::Float64
end

LearningProblem(
    nvar::Int, ϵ::Float64, θ::Float64, δ::Float64
) = LearningProblem(
    nvar, System[], ϵ, θ, δ, Float64[], _VT_[],
    eps(1.0), eps(1.0), -eps(1.0), eps(1.0)
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

function add_point!(prob::LearningProblem, point::_VT_)
    np = norm(point, Inf)
    if np < prob.tol_norm*prob.nvar
        error(string("Point norm close to zero: ", np))
    end
    push!(prob.points, point/np)
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

function _make_verifs(prob)
    Q = length(prob.systems)
    verifs_pos = Vector{VerifierPos}(undef, Q)
    verifs_lie = Vector{VerifierLie}(undef, Q)
    for (q, system) in enumerate(prob.systems)
        verifs_pos[q] = VerifierPos(prob.nvar, system.domain)
        verifs_lie[q] = VerifierLie(prob.nvar, system.domain, system.A)
    end
    return verifs_pos, verifs_lie
end

_init_trace() = (
    vecs_list=Vector{_VT_}[],
    witnesses_list=Vector{Witness}[]
)

function _verify(verifs_pos, verifs_lie, vecs, tol_pos, tol_lie, solver)
    print("Verify pos... ")
    x, val, q = verify(verifs_pos, vecs, solver)
    if val < tol_pos
        println("CE found: ", x, ", ", val, ", ", q)
        return x
    else
        println("No CE found: ", val)
    end
    print("Verify lie... ")
    x, val, q = verify(verifs_lie, vecs, solver)
    if val > tol_lie
        println("CE found: ", x, ", ", val, ", ", q)
        return x
    else
        println("No CE found: ", val)
    end
    return Float64[]
end

function learn_lyapunov(prob::LearningProblem, iter_max, solver)
    vecsgen = VecsGenerator(prob.nvar, prob.ϵ, prob.θ, prob.δ, prob.Gs)

    trace_out = _init_trace()

    witnesses_init = Witness[]
    for point in prob.points
        wit = _add_vecs_point!(vecsgen, prob.systems, point)
        push!(witnesses_init, wit)
    end
    push!(trace_out.witnesses_list, witnesses_init)

    verifs_pos, verifs_lie = _make_verifs(prob)

    iter = 0

    while true
        iter += 1
        if iter > iter_max
            println(string("Max iter exceeded: ", iter))
            return _VT_[], -Inf, trace_out
        end

        flag = check_feasibility(vecsgen, solver)
        if !flag
            println(string(
                "System does not admit a Lyapunov function with parameters: ",
                "ϵ: ", prob.ϵ, ", θ: ", prob.θ, ", δ: ", prob.δ
            ))
            return _VT_[], -Inf, trace_out
        end

        vecs, r = compute_vecs(vecsgen, solver)
        push!(trace_out.vecs_list, vecs)
        if r < prob.tol_rad
            println(string("Satisfiability radius too small: ", r))
            return _VT_[], -Inf, trace_out
        end

        x = _verify(
            verifs_pos, verifs_lie, vecs, prob.tol_pos, prob.tol_lie, solver
        )
        if isempty(x)
            println("No CE found")
            println("Valid CLF: terminated")
            return vecs, r, trace_out
        end

        point = x/norm(x, Inf)
        wit = _add_vecs_point!(vecsgen, prob.systems, point)
        push!(trace_out.witnesses_list, [wit])
    end
end