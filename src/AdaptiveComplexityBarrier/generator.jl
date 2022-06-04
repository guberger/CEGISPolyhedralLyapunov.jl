struct NegEvidence
    point::_VT_
    npoint::Float64
end

struct PosEvidence
    point::_VT_
    npoint::Float64
end

struct LieEvidence
    point1::_VT_
    point2::_VT_
    loc2::Int
    npoint1::Float64
    npoint2::Float64
    nA::Float64
end

struct Witness
    loc::Int
    neg_evids::Vector{NegEvidence}
    pos_evids::Vector{PosEvidence}
    lie_evids::Vector{LieEvidence}
end

Witness(loc::Int, neg_evids::Vector{NegEvidence}) = Witness(
    loc, neg_evids, PosEvidence[], LieEvidence[]
)

function add_evidence!(wit::Witness, pos_evid::PosEvidence)
    push!(wit.pos_evids, pos_evid)
end

function add_evidence!(wit::Witness, lie_evid::LieEvidence)
    push!(wit.lie_evids, lie_evid)
end

function add_evidence_pos!(wit::Witness, point, npoint)
    add_evidence!(wit, PosEvidence(point, npoint))
end

function add_evidence_lie!(
        wit::Witness, point1, point2, loc2, npoint1, npoint2, nA
    )
    add_evidence!(wit, LieEvidence(point1, point2, loc2, npoint1, npoint2, nA))
end

struct Generator
    nvar::Int
    nloc::Int
    witnesses::Vector{Witness}
end

Generator(nvar::Int, nloc::Int) = Generator(nvar, nloc, Witness[])

function add_witness!(gen::Generator, wit::Witness)
    push!(gen.witnesses, wit)
end

## Compute afs

struct _AF
    lin::Vector{VariableRef}
    con::VariableRef
end
_eval(af::_AF, point) = dot(af.lin, point) + af.con

function _add_vars!(model, nvar, nwit)
    afs = Vector{_AF}(undef, nwit)
    for i = 1:nwit
        lin = @variable(model, [1:nvar], lower_bound=-1, upper_bound=1)
        con = @variable(model, lower_bound=-1, upper_bound=1)
        afs[i] = _AF(lin, con)
        #new:
        alin = @variable(model, [1:nvar], lower_bound=0, upper_bound=1)
        @constraint(model, -a .≤ alin)
        @constraint(model, +a .≤ alin)
        @constraint(model, sum(alin) + con ≤ 1)
        @constraint(model, sum(alin) - con ≤ 1) # end new
    end
    r = @variable(model, upper_bound=2)
    return afs, r
end

function _add_neg_constr!(model, afs, r, i, point, α, β, off)
    @constraint(model, α*_eval(afs[i], point) + β*r + off ≤ 0)
end

function _add_pos_constr!(model, afs, r, i, point, α, β, off)
    @constraint(model, α*_eval(afs[i], point) - β*r - off ≥ 0)
end

function _add_lie_constr(
        model, afs, r, i, j1, j2, point1, point2, α, β, γ, off
    )
    @constraint(model,
        α*(_eval(afs[j2], point2) - _eval(afs[j1], point1)) + β*r + off -
            γ*(_eval(afs[i], point1) - _eval(afs[j1], point1)) ≤ 0
    )
end

function _make_loc_dict(witnesses::Vector{Witness})
    loc_dict = Dict{Int,Vector{Int}}()
    for (i, wit) in enumerate(witnesses)
        i_list = get(loc_dict, wit.loc, Int[])
        push!(i_list, i)
    end
    return loc_dict
end

abstract type GeneratorProblem end

function _compute_afs(prob::GeneratorProblem, nvar, witnesses, solver)
    model = Model(solver)
    afs, r = _add_vars!(model, nvar, length(witnesses))
    loc_dict = _make_loc_dict(witnesses)

    for (i, wit) in enumerate(witnesses)
        for negevid in wit.neg_evids
            _add_neg_constr_prob!(prob, model, afs, r, i, negevid)
        end
        for posevid in wit.pos_evids
            _add_pos_constr_prob!(prob, model, afs, r, i, posevid)
        end
        j1_list = loc_dict[wit.loc]
        for lieevid in wit.lie_evids
            j2_list = loc_dict[lieevid.loc2]
            for (j1, j2) in Iterators.product(j1_list, j2_list)
                _add_lie_constr_prob!(prob, model, afs, r, i, j1, j2, lieevid)
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

    return [AffForm(value.(af.lin), value(af.con)) for af in afs], value(r)
end

## Feasibility

struct GeneratorFeasibility <: GeneratorProblem
    ϵ::Float64
    θ::Float64
    δ::Float64
end

function _add_neg_constr_prob!(
        prob::GeneratorFeasibility, model, afs, r, i, negevid
    )
    point = negevid.point
    off = negevid.npoint/prob.ϵ
    β = negevid.npoint
    _add_neg_constr!(model, afs, r, i, point, 1, β, off)
end

function _add_pos_constr_prob!(
        prob::GeneratorFeasibility, model, afs, r, i, posevid
    )
    point = posevid.point
    off = posevid.npoint/prob.ϵ
    β = posevid.npoint
    _add_pos_constr!(model, afs, r, i, point, 1, β, off)
end

function _add_lie_constr_prob!(
        prob::GeneratorFeasibility, model, afs, r, i, j1, j2, lieevid
    )
    point1 = lieevid.point1
    point2 = lieevid.point2
    off = lieevid.npoint1*prob.δ
    β = lieevid.npoint1
    γ::Float64 = i == j1 ? 0.0 : 1/prob.θ
    _add_lie_constr(model, afs, r, i, j1, j2, point1, point2, 1, β, γ, off)
end

function compute_afs_feasibility(
        gen::Generator, ϵ::Float64, θ::Float64, δ::Float64, solver
    )
    prob = GeneratorFeasibility(ϵ, θ, δ)
    return _compute_afs(prob, gen.nvar, gen.witnesses, solver)
end

## Chebyshev

struct GeneratorChebyshev <: GeneratorProblem
    G::Float64
end

function _add_neg_constr_prob!(
        ::GeneratorChebyshev, model, afs, r, i, negevid
    )
    point = negevid.point
    β = negevid.npoint
    _add_neg_constr!(model, afs, r, i, point, 1, β, 0)
end

function _add_pos_constr_prob!(
        ::GeneratorChebyshev, model, afs, r, i, posevid
    )
    point = posevid.point
    β = posevid.npoint
    _add_pos_constr!(model, afs, r, i, point, 1, β, 0)
end

function _add_lie_constr_prob!(
        prob::GeneratorChebyshev, model, afs, r, i, j1, j2, lieevid
    )
    point1 = lieevid.point1
    point2 = lieevid.point2
    G::Float64 = i == j1 ? 0.0 : prob.G
    α = 1
    β = lieevid.npoint1*(G + abs(G - α)) + lieevid.npoint2
    _add_lie_constr(model, afs, r, i, j1, j2, point1, point2, α, β, G, 0)
end

function compute_afs_chebyshev(gen::Generator, G::Float64, solver)
    prob = GeneratorChebyshev(G)
    return _compute_afs(prob, gen.nvar, gen.witnesses, solver)
end

## Witness

struct GeneratorWitness <: GeneratorProblem
    G::Float64
end

function _add_neg_constr_prob!(
        ::GeneratorWitness, model, afs, r, i, negevid
    )
    point = negevid.point
    β = negevid.npoint
    _add_neg_constr!(model, afs, r, i, point, 1, β, 0)
end

function _add_pos_constr_prob!(
        ::GeneratorWitness, model, afs, r, i, posevid
    )
    point = posevid.point
    β = posevid.npoint
    _add_pos_constr!(model, afs, r, i, point, 1, β, 0)
end

function _add_lie_constr_prob!(
        prob::GeneratorWitness, model, afs, r, i, j1, j2, lieevid
    )
    point1 = lieevid.point1
    point2 = lieevid.point2
    G::Float64 = i == j1 ? 0.0 : prob.G
    α = 1
    β = lieevid.npoint1*(G + abs(G - α)) + lieevid.nA*lieevid.npoint1
    _add_lie_constr(model, afs, r, i, j1, j2, point1, point2, α, β, G, 0)
end

function compute_afs_witness(gen::Generator, G::Float64, solver)
    prob = GeneratorWitness(G)
    return _compute_afs(prob, gen.nvar, gen.witnesses, solver)
end