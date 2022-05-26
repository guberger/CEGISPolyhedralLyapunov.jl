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

struct GeneratingProblem
    nvar::Int
    witnesses::Vector{Witness}
end

GeneratingProblem(nvar::Int) = GeneratingProblem(nvar, Witness[])

function add_witness!(gen::GeneratingProblem, wit::Witness)
    push!(gen.witnesses, wit)
end

struct PosConstraint
    point::_VT_
    i::Int
    off::Float64
    α::Float64 # α*r
end

struct LieConstraint
    point::_VT_
    deriv::_VT_
    i::Int
    j::Int
    k::Int
    off::Float64
    α::Float64 # α*r
    β::Float64 # β*G*r
    γ::Float64 # γ*G*(ci*point - cj*point)
end

## Compute vecs r

function _add_vars_vecs_r!(model, nvar, nwit)
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

function _add_pos_constr_vecs_r!(model, vecs, r, poscon)
    point = poscon.point
    vec = vecs[poscon.i]
    off = poscon.off
    α = poscon.α
    @constraint(model, dot(point, vec) ≥ α*r + off)
end

function _add_lie_constr_vecs_r!(model, vecs, r, Gs, liecon)
    point = liecon.point
    deriv = liecon.deriv
    vec1 = vecs[liecon.i]
    vec2 = vecs[liecon.j]
    G = liecon.k == 0 ? 0.0 : Gs[liecon.k]
    off = liecon.off
    α = liecon.α
    β = liecon.β
    γ = liecon.γ
    @constraint(
        model,
        dot(deriv, vec2) ≤ γ*G*dot(point, vec1 - vec2) - α*r - β*G*r - off
    )
end

function _compute_vecs_r(nvar, nwit, pos_constrs, lie_constrs, Gs, solver)
    model = Model(solver)
    vecs, r = _add_vars_vecs_r!(model, nvar, nwit)

    for poscon in pos_constrs
        _add_pos_constr_vecs_r!(model, vecs, r, poscon)
    end

    for liecon in lie_constrs
        _add_lie_constr_vecs_r!(model, vecs, r, Gs, liecon)
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

function _make_pos_constrs_feasibility(witnesses, ϵ)
    pos_constrs = PosConstraint[]
    for (i, wit) in enumerate(witnesses)
        for posevid in wit.pos_evids
            point = posevid.point
            off = posevid.npoint/ϵ
            α = posevid.npoint
            push!(pos_constrs, PosConstraint(point, i, off, α))
        end
    end
    return pos_constrs
end

function _make_lie_constrs_feasibility(witnesses, δ)
    lie_constrs = LieConstraint[]
    for (i, wit) in enumerate(witnesses)
        for lieevid in wit.lie_evids
            point = lieevid.point
            deriv = lieevid.deriv
            off = lieevid.npoint*lieevid.nA*δ
            α = lieevid.npoint*lieevid.nA
            β = 0.0
            γ = lieevid.nA
            for j = 1:length(witnesses)
                push!(
                    lie_constrs,
                    LieConstraint(point, deriv, i, j, 1, off, α, β, γ)
                )
            end
        end
    end
    return lie_constrs
end

function compute_feasibility(
        gen::GeneratingProblem, ϵ::Float64, θ::Float64, δ::Float64, solver
    )
    pos_constrs = _make_pos_constrs_feasibility(gen.witnesses, ϵ)
    lie_constrs = _make_lie_constrs_feasibility(gen.witnesses, δ)
    Gs = [1/θ]

    return _compute_vecs_r(
        gen.nvar, length(gen.witnesses), pos_constrs, lie_constrs, Gs, solver
    )
end

## Heuristic

function _make_pos_constrs_heuristic(witnesses)
    pos_constrs = PosConstraint[]
    for (i, wit) in enumerate(witnesses)
        for posevid in wit.pos_evids
            point = posevid.point
            off = 0.0
            α = posevid.npoint
            push!(pos_constrs, PosConstraint(point, i, off, α))
        end
    end
    return pos_constrs
end

function _make_lie_constrs_heuristic(witnesses)
    lie_constrs = LieConstraint[]
    for (i, wit) in enumerate(witnesses)
        for lieevid in wit.lie_evids
            point = lieevid.point
            deriv = lieevid.deriv
            off = 0.0
            α = lieevid.nderiv
            β = 2.0
            γ = 1.0
            for j = 1:length(witnesses)
                k = i == j ? 0 : 1
                push!(
                    lie_constrs,
                    LieConstraint(point, deriv, i, j, k, off, α, β, γ)
                )
            end
        end
    end
    return lie_constrs
end

function compute_vecs_r_heuristic(gen::GeneratingProblem, G::Float64, solver)
    pos_constrs = _make_pos_constrs_heuristic(gen.witnesses)
    lie_constrs = _make_lie_constrs_heuristic(gen.witnesses)
    Gs = [G]

    return _compute_vecs_r(
        gen.nvar, length(gen.witnesses), pos_constrs, lie_constrs, Gs, solver
    )
end

## Alternating

function _make_pos_constrs_alternating(witnesses)
    pos_constrs = PosConstraint[]
    for (i, wit) in enumerate(witnesses)
        for posevid in wit.pos_evids
            point = posevid.point
            off = 0.0
            α = posevid.npoint
            push!(pos_constrs, PosConstraint(point, i, off, α))
        end
    end
    return pos_constrs
end

function _make_lie_constrs_alternating(witnesses)
    lie_constrs = LieConstraint[]
    runner = 1
    for (i, wit) in enumerate(witnesses)
        for lieevid in wit.lie_evids
            point = lieevid.point
            deriv = lieevid.deriv
            off = 0.0
            α = lieevid.npoint*lieevid.nA
            β = 2*lieevid.npoint*lieevid.nA
            γ = lieevid.nA
            for j = 1:length(witnesses)
                k = i == j ? 0 : runner
                push!(
                    lie_constrs,
                    LieConstraint(point, deriv, i, j, k, off, α, β, γ)
                )
                runner += (i != j)
            end
        end
    end
    return lie_constrs, runner
end

function compute_vecs_r_alternating(
        gen::GeneratingProblem, G::Float64, niter::Int, solver
    )
    pos_constrs = _make_pos_constrs_alternating(gen.witnesses)
    lie_constrs = _make_lie_constrs_alternating(gen.witnesses)

    Gs = fill(G, nG)

    
end
=#

#=
function _add_constrs_vecs_rad!(model, vecs, r, i, wit, ϵ, Gs)
    for poscon in wit.pos_evids
        point = poscon.point
        α = poscon.npoint*r # new Eccentricity V1
        # α = poscon.npoint/ϵ # old and new Eccentricity V2
        @constraint(model, dot(point, vecs[i]) ≥ α)
    end
    for liecon in wit.lie_evids
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

function compute_vecs(gen::GeneratingProblem, solver)
    vecsopt = Vector{Float64}[]
    ropt = -Inf
    for (k, G) in enumerate(gen.Gs)
        r = gen.rs[k]
        r < ropt && continue
        vecs, r = _compute_vecs(gen, G, solver)
        gen.rs[k] = r
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
=#