using LinearAlgebra
using JuMP
const _RSC_ = JuMP.MathOptInterface.ResultStatusCode
const _TSC_ = JuMP.MathOptInterface.TerminationStatusCode

_VT_ = Vector{Float64}

struct PosEvidence
    point::_VT_
    npoint::Float64
end

struct LieEvidence
    point::_VT_
    deriv::_VT_
    npoint::Float64
    nderiv::Float64
    nA::Float64
end

struct Witness
    pos_evids::Vector{PosEvidence}
    lie_evids::Vector{LieEvidence}
end

Witness() = Witness(PosEvidence[], LieEvidence[])

function add_evidence!(wit::Witness, pos_evid::PosEvidence)
    push!(wit.pos_evids, pos_evid)
end

function add_evidence!(wit::Witness, lie_evid::LieEvidence)
    push!(wit.lie_evids, lie_evid)
end

struct Generator
    nvar::Int
    witnesses::Vector{Witness}
end

Generator(nvar::Int) = Generator(nvar, Witness[])

function add_witness!(gen::Generator, wit::Witness)
    push!(gen.witnesses, wit)
end

## Compute vecs

function _add_vars!(model, nvar, nwit)
    vecs = Vector{Vector{VariableRef}}(undef, nwit)
    for i = 1:nwit
        vec = @variable(model, [1:nvar], lower_bound=-1, upper_bound=1)
        vecs[i] = vec
        #new:
        avec = @variable(model, [1:nvar], lower_bound=0, upper_bound=1)
        @constraint(model, -vec .≤ avec)
        @constraint(model, +vec .≤ avec)
        @constraint(model, sum(avec) ≤ 1) # end new
    end
    r = @variable(model, upper_bound=2)
    return vecs, r
end

function _add_pos_constr!(model, vecs, r, i, point, α, β, off)
    @constraint(model, α*dot(point, vecs[i]) ≥ β*r + off)
end

function _add_lie_constr(model, vecs, r, i, j, point, deriv, α, β, γ, off)
    @constraint(model,
        α*dot(deriv, vecs[j]) + β*r + off ≤ γ*dot(point, vecs[i] - vecs[j])
    )
end

abstract type GeneratorProblem end

function _compute_vecs(prob::GeneratorProblem, nvar, witnesses, solver)
    model = Model(solver)
    vecs, r = _add_vars!(model, nvar, length(witnesses))

    for (i, wit) in enumerate(witnesses)
        for posevid in wit.pos_evids
            _add_pos_constr_prob!(prob, model, vecs, r, i, posevid)
        end
        for lieevid in wit.lie_evids
            for j = 1:length(witnesses)
                _add_lie_constr_prob!(prob, model, vecs, r, i, j, lieevid)
            end
        end
    end

    @objective(model, Max, r)

    optimize!(model)

    if !(primal_status(model) == _RSC_(1) &&
            termination_status(model) == _TSC_(1))
        error(string(
            "Generator: not optimal: ",
            primal_status(model), " ",
            dual_status(model), " ",
            termination_status(model)
        ))
    end

    return [value.(vec) for vec in vecs], value(r)
end

## Feasibility

struct GeneratorFeasibility <: GeneratorProblem
    ϵ::Float64
    θ::Float64
    δ::Float64
end

function _add_pos_constr_prob!(
        prob::GeneratorFeasibility, model, vecs, r, i, posevid
    )
    point = posevid.point
    off = posevid.npoint/prob.ϵ
    β = posevid.npoint
    _add_pos_constr!(model, vecs, r, i, point, 1.0, β, off)
end

function _add_lie_constr_prob!(
        prob::GeneratorFeasibility, model, vecs, r, i, j, lieevid
    )
    point = lieevid.point
    deriv = lieevid.deriv
    off = lieevid.npoint*prob.δ
    α = 1/lieevid.nA
    β = lieevid.npoint
    γ::Float64 = i == j ? 0.0 : 1/prob.θ
    _add_lie_constr(model, vecs, r, i, j, point, deriv, α, β, γ, off)
end

function compute_vecs_feasibility(
        gen::Generator, ϵ::Float64, θ::Float64, δ::Float64, solver
    )
    prob = GeneratorFeasibility(ϵ, θ, δ)
    return _compute_vecs(prob, gen.nvar, gen.witnesses, solver)
end

## Heuristic

struct GeneratorHeuristic <: GeneratorProblem
    G::Float64
end

function _add_pos_constr_prob!(
        prob::GeneratorHeuristic, model, vecs, r, i, posevid
    )
    point = posevid.point
    β = posevid.npoint
    _add_pos_constr!(model, vecs, r, i, point, 1.0, β, 0.0)
end

function _add_lie_constr_prob!(
        prob::GeneratorHeuristic, model, vecs, r, i, j, lieevid
    )
    point = lieevid.point
    deriv = lieevid.deriv
    H::Float64 = i == j ? 0.0 : prob.G
    β = 2*lieevid.npoint*H + lieevid.nderiv
    γ::Float64 = i == j ? 0.0 : prob.G
    _add_lie_constr(model, vecs, r, i, j, point, deriv, 1.0, β, γ, 0.0)
end

function compute_vecs_heuristic(gen::Generator, G::Float64, solver)
    prob = GeneratorHeuristic(G)
    return _compute_vecs(prob, gen.nvar, gen.witnesses, solver)
end