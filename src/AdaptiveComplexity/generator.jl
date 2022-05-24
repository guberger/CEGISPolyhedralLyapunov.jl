using LinearAlgebra
using JuMP
const _RSC_ = JuMP.MathOptInterface.ResultStatusCode
const _TSC_ = JuMP.MathOptInterface.TerminationStatusCode

_VT_ = Vector{Float64}

struct PosConstraint
    point::_VT_
    npoint::Float64
end

struct LieConstraint
    point::_VT_
    deriv::_VT_
    npoint::Float64
    nderiv::Float64
    nA::Float64
end

struct Witness
    pos_constrs::Vector{PosConstraint}
    lie_constrs::Vector{LieConstraint}
end

mutable struct VecsGenerator
    nvar::Int
    witnesses::Vector{Witness}
    Gs::Vector{Float64}
    rs::Vector{Float64}
end

VecsGenerator(nvar::Int, Gs::Vector{Float64}) =
    VecsGenerator(nvar, Witness[], Gs, fill(Inf, length(Gs)))

function add_witness!(vecsgen::VecsGenerator, wit::Witness)
    push!(vecsgen.witnesses, wit)
end

function _add_vars!(model, nvar, nvec)
    vecs = [
        @variable(model, [1:nvar], lower_bound=-1, upper_bound=1) for i = 1:nvec
    ]
    # new:
    avecs = [
        @variable(model, [1:nvar], lower_bound=0, upper_bound=1) for i = 1:nvec
    ]
    for i = 1:nvec
        @constraint(model, -vecs[i] .≤ avecs[i])
        @constraint(model, +vecs[i] .≤ avecs[i])
        @constraint(model, sum(avecs[i]) ≤ 1)
    end # end new
    return vecs
end

function _add_constrs_compute_feasibility!(model, vecs, r, i, wit, ϵ, θ)
    for poscon in wit.pos_constrs
        point = poscon.point
        @constraint(model, dot(point, vecs[i]) ≥ poscon.npoint/ϵ)
    end
    for liecon in wit.lie_constrs
        point = liecon.point
        deriv = liecon.deriv
        α = liecon.nA*liecon.npoint*r
        @constraint(model, dot(deriv, vecs[i]) + α ≤ 0)
        G = liecon.nA/θ
        for j = 1:length(vecs)
            j == i && continue
            @constraint(
                model,
                dot(deriv, vecs[j]) - G*dot(point, vecs[i] - vecs[j]) + α ≤ 0
            )
        end
    end
end

function compute_feasibility(
        vecsgen::VecsGenerator, ϵ::Float64, θ::Float64, solver
    )
    model = Model(solver)
    vecs = _add_vars!(model, vecsgen.nvar, length(vecsgen.witnesses))
    r = @variable(model, upper_bound=2)

    for (i, wit) in enumerate(vecsgen.witnesses)
        _add_constrs_compute_feasibility!(model, vecs, r, i, wit, ϵ, θ)
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

    return value(r)
end

function _add_constrs_compute_vecs!(model, vecs, r, i, wit, ϵ, G)
    for poscon in wit.pos_constrs
        point = poscon.point
        α = poscon.npoint*r # new Eccentricity V1
        # α = poscon.npoint/ϵ # old and new Eccentricity V2
        @constraint(model, dot(point, vecs[i]) ≥ α)
    end
    for liecon in wit.lie_constrs
        point = liecon.point
        deriv = liecon.deriv
        α = liecon.nderiv*r # new Deriv V1
        # α = liecon.nA*liecon.npoint*r # new Deriv V2
        @constraint(model, dot(deriv, vecs[i]) + α ≤ 0)
        β = α + 2*G*liecon.npoint*r # new
        # β = α # old
        for j = 1:length(vecs)
            j == i && continue
            @constraint(
                model,
                dot(deriv, vecs[j]) - G*dot(point, vecs[i] - vecs[j]) + β ≤ 0
            )
        end
    end
end

function _compute_vecs(vecsgen::VecsGenerator, G::Float64, solver)
    model = Model(solver)
    vecs = _add_vars!(model, vecsgen.nvar, length(vecsgen.witnesses))
    r = @variable(model, upper_bound=2)

    for (i, wit) in enumerate(vecsgen.witnesses)
        _add_constrs_compute_vecs!(model, vecs, r, i, wit, 0.0, G)
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