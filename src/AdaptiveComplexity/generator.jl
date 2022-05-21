using LinearAlgebra
using JuMP
const _RSC_ = JuMP.MathOptInterface.ResultStatusCode
const _TSC_ = JuMP.MathOptInterface.TerminationStatusCode

_VT_ = Vector{Float64}

struct Witness
    point::_VT_
    derivs::Vector{_VT_}
end

Witness(point::_VT_) = Witness(point, _VT_[])

mutable struct VecsGenerator
    nvar::Int
    nvec::Int
    witnesses_map::Vector{Tuple{Int,Witness}}
    ϵ::Float64
    θ::Float64
    δ::Float64
    Gs::Vector{Float64}
    rs::Vector{Float64}
end

VecsGenerator(
    nvar::Int, ϵ::Float64, θ::Float64, δ::Float64, Gs::Vector{Float64}
) = VecsGenerator(
    nvar, 0, Tuple{Int,Witness}[], ϵ, θ, δ, Gs, fill(Inf, length(Gs))
)

function add_deriv!(wit::Witness, deriv::_VT_)
    push!(wit.derivs, deriv)
end

function add_vec!(vecsgen::VecsGenerator)
    vecsgen.nvec += 1
    return vecsgen.nvec
end

function add_witness!(vecsgen::VecsGenerator, i::Int, wit::Witness)
    push!(vecsgen.witnesses_map, (i, wit))
end

function _add_vars!(model, nvar, nvec)
    vecs = [
        @variable(model, [1:nvar], lower_bound=-1, upper_bound=1) for i = 1:nvec
    ]
    avecs = [
        @variable(model, [1:nvar], lower_bound=0, upper_bound=1) for i = 1:nvec
    ]
    for i = 1:nvec
        @constraint(model, -vecs[i] .≤ avecs[i])
        @constraint(model, +vecs[i] .≤ avecs[i])
        @constraint(model, sum(avecs[i]) ≤ 1)
    end
    return vecs
end

function _add_constr_pos!(model, vec, wit, β)
    @constraint(model, dot(wit.point, vec) ≥ β*norm(wit.point, Inf))
end

function _add_constr_lie!(model, vec, vec2, wit, β, γ)
    for deriv in wit.derivs
        @constraint(
            model,
            dot(deriv, vec2) -
                γ*dot(wit.point, vec - vec2) +
                β*norm(wit.point, Inf) ≤ 0
        )
    end
end

function check_feasibility(vecsgen::VecsGenerator, solver)
    if iszero(vecsgen.nvec)
        return true
    end

    model = Model(solver)
    vecs = _add_vars!(model, vecsgen.nvar, vecsgen.nvec)
    βpos = 1/vecsgen.ϵ
    βlie = vecsgen.δ
    γ = 1/vecsgen.θ

    for (i, wit) in vecsgen.witnesses_map
        vec = vecs[i]
        _add_constr_pos!(model, vec, wit, βpos)
        _add_constr_lie!(model, vec, vec, wit, βlie, 0)
        for j = 1:vecsgen.nvec
            j == i && continue
            vec2 = vecs[j]
            _add_constr_lie!(model, vec, vec2, wit, βlie, γ)
        end
    end

    optimize!(model)

    if primal_status(model) == _RSC_(1) && termination_status(model) == _TSC_(1)
        return true
    elseif primal_status(model) == _RSC_(0) && termination_status(model) == _TSC_(2)
        return false
    else
        error(string(
            "Generator: neither feasible or infeasible: ",
            primal_status(model), " ",
            dual_status(model), " ",
            termination_status(model)
        ))
    end
end

function _compute_vecs(vecsgen::VecsGenerator, G, solver)
    model = Model(solver)
    vecs = _add_vars!(model, vecsgen.nvar, vecsgen.nvec)
    r = @variable(model, upper_bound=2)

    for (i, wit) in vecsgen.witnesses_map
        vec = vecs[i]
        _add_constr_pos!(model, vec, wit, r)
        _add_constr_lie!(model, vec, vec, wit, r, 0)
        for j = 1:vecsgen.nvec
            j == i && continue
            vec2 = vecs[j]
            _add_constr_lie!(model, vec, vec2, wit, (2*G + 1)*r, G)
        end
    end

    @objective(model, Max, r)

    optimize!(model)

    if !(primal_status(model) == _RSC_(1) && termination_status(model) == _TSC_(1))
        error(string(
            "Generator: not optimal: ",
            primal_status(model), " ",
            dual_status(model), " ",
            termination_status(model)
        ))
    end

    return [value.(vec) for vec in vecs], value(r)
end

function compute_vecs(vecsgen::VecsGenerator, solver)
    vecsopt = Vector{Float64}[]
    ropt = -Inf
    for (k, G) in enumerate(vecsgen.Gs)
        r = vecsgen.rs[k]
        r < ropt && continue
        vecs, r = _compute_vecs(vecsgen, G, solver)
        vecsgen.rs[k] = r
        if r > ropt
            ropt = r
            vecsopt = vecs
        end
    end
    if isinf(ropt)
        error(string("Generator: inf radius: ", ropt))
    end
    return vecsopt, ropt
end