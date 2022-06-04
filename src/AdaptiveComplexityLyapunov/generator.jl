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

function add_evidence_pos!(wit::Witness, point, npoint)
    add_evidence!(wit, PosEvidence(point, npoint))
end

function add_evidence_lie!(wit::Witness, point, deriv, npoint, nderiv, nA)
    add_evidence!(wit, LieEvidence(point, deriv, npoint, nderiv, nA))
end

struct Generator
    nvar::Int
    witnesses::Vector{Witness}
end

Generator(nvar::Int) = Generator(nvar, Witness[])

function add_witness!(gen::Generator, wit::Witness)
    push!(gen.witnesses, wit)
end

## Compute lfs

struct _LF
    lin::Vector{VariableRef}
end
_eval(lf::_LF, point) = dot(point, lf.lin)

function _add_vars!(model, nvar, nwit)
    lfs = Vector{_LF}(undef, nwit)
    for i = 1:nwit
        lin = @variable(model, [1:nvar], lower_bound=-1, upper_bound=1)
        lfs[i] = _LF(lin)
        #new:
        alin = @variable(model, [1:nvar], lower_bound=0, upper_bound=1)
        @constraint(model, -lin .≤ alin)
        @constraint(model, +lin .≤ alin)
        @constraint(model, sum(alin) ≤ 1) # end new
    end
    r = @variable(model, upper_bound=2)
    return lfs, r
end

function _add_pos_constr!(model, lfs, r, i, point, α, β, off)
    @constraint(model, α*_eval(lfs[i], point) ≥ β*r + off)
end

function _add_lie_constr(model, lfs, r, i, j, point, deriv, α, β, γ, off)
    @constraint(model,
        α*_eval(lfs[j], deriv) + β*r + off -
            γ*(_eval(lfs[i], point) - _eval(lfs[j], point)) ≤ 0
    )
end

abstract type GeneratorProblem end

function _compute_lfs(prob::GeneratorProblem, nvar, witnesses, solver)
    model = Model(solver)
    lfs, r = _add_vars!(model, nvar, length(witnesses))

    for (i, wit) in enumerate(witnesses)
        for posevid in wit.pos_evids
            _add_pos_constr_prob!(prob, model, lfs, r, i, posevid)
        end
        for lieevid in wit.lie_evids
            for j = 1:length(witnesses)
                _add_lie_constr_prob!(prob, model, lfs, r, i, j, lieevid)
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

    return [LinForm(value.(lf.lin)) for lf in lfs], value(r)
end

## Feasibility

struct GeneratorFeasibility <: GeneratorProblem
    ϵ::Float64
    θ::Float64
    δ::Float64
end

function _add_pos_constr_prob!(
        prob::GeneratorFeasibility, model, lfs, r, i, posevid
    )
    point = posevid.point
    off = posevid.npoint/prob.ϵ
    β = posevid.npoint
    _add_pos_constr!(model, lfs, r, i, point, 1, β, off)
end

function _add_lie_constr_prob!(
        prob::GeneratorFeasibility, model, lfs, r, i, j, lieevid
    )
    point = lieevid.point
    deriv = lieevid.deriv
    off = lieevid.npoint*prob.δ
    α = 1/lieevid.nA # TODO: update with α = 1
    β = lieevid.npoint
    γ::Float64 = i == j ? 0.0 : 1/prob.θ
    _add_lie_constr(model, lfs, r, i, j, point, deriv, α, β, γ, off)
end

function compute_lfs_feasibility(
        gen::Generator, ϵ::Float64, θ::Float64, δ::Float64, solver
    )
    prob = GeneratorFeasibility(ϵ, θ, δ)
    return _compute_lfs(prob, gen.nvar, gen.witnesses, solver)
end

## Chebyshev

struct GeneratorChebyshev <: GeneratorProblem
    G::Float64
end

function _add_pos_constr_prob!(
        ::GeneratorChebyshev, model, lfs, r, i, posevid
    )
    point = posevid.point
    β = posevid.npoint
    _add_pos_constr!(model, lfs, r, i, point, 1, β, 0)
end

function _add_lie_constr_prob!(
        prob::GeneratorChebyshev, model, lfs, r, i, j, lieevid
    )
    point = lieevid.point
    deriv = lieevid.deriv
    G::Float64 = i == j ? 0.0 : prob.G
    β = 2*lieevid.npoint*G + lieevid.nderiv
    _add_lie_constr(model, lfs, r, i, j, point, deriv, 1, β, G, 0)
end

function compute_lfs_chebyshev(gen::Generator, G::Float64, solver)
    prob = GeneratorChebyshev(G)
    return _compute_lfs(prob, gen.nvar, gen.witnesses, solver)
end

## Witness

struct GeneratorWitness <: GeneratorProblem
    G::Float64
end

function _add_pos_constr_prob!(
        ::GeneratorWitness, model, lfs, r, i, posevid
    )
    point = posevid.point
    β = posevid.npoint
    _add_pos_constr!(model, lfs, r, i, point, 1, β, 0)
end

function _add_lie_constr_prob!(
        prob::GeneratorWitness, model, lfs, r, i, j, lieevid
    )
    point = lieevid.point
    deriv = lieevid.deriv
    G::Float64 = i == j ? 0.0 : prob.G
    β = 2*lieevid.npoint*G + lieevid.nA*lieevid.npoint
    _add_lie_constr(model, lfs, r, i, j, point, deriv, 1, β, G, 0)
end

function compute_lfs_witness(gen::Generator, G::Float64, solver)
    prob = GeneratorWitness(G)
    return _compute_lfs(prob, gen.nvar, gen.witnesses, solver)
end