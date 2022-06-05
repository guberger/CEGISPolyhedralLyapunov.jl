struct PosEvidence
    point::_VT_
    loc::Int
    npoint::Float64
end

struct LieEvidence
    point1::_VT_
    loc1::Int
    point2::_VT_
    loc2::Int
    npoint1::Float64
    npoint2::Float64
    ndiff::Float64
    nA::Float64
    nD::Float64
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

function add_evidence_pos!(wit::Witness, point, loc, npoint)
    add_evidence!(wit, PosEvidence(point, loc, npoint))
end

function add_evidence_lie!(
        wit::Witness,
        point1, loc1, point2, loc2, npoint1, npoint2, ndiff, nA, nD
    )
    add_evidence!(wit, LieEvidence(
        point1, loc1, point2, loc2, npoint1, npoint2, ndiff, nA, nD
    ))
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

function _add_pos_constr!(model, lfs, r, i, point, α, β)
    @constraint(model, _eval(lfs[i], point) ≥ α*r + β)
end

function _add_lie_constr(model, lfs, r, i1, i2, point1, point2, α, β)
    @constraint(model,
        _eval(lfs[i2], point2) + α*r + β ≤ _eval(lfs[i1], point1)
    )
end

_value(lf::_LF) = LinForm(value.(lf.lin))

function _make_loc_map(nloc, witnesses)
    loc_map = [BitSet() for loc = 1:nloc]
    for (i, wit) in enumerate(witnesses)
        for posevid in wit.pos_evids
            push!(loc_map[posevid.loc], i)
        end
        for lievid in wit.lie_evids
            push!(loc_map[lievid.loc1], i)
        end
    end
    return loc_map
end

abstract type GeneratorProblem end

function _compute_polyf(prob::GeneratorProblem, nvar, nloc, witnesses, solver)
    model = Model(solver)
    lfs, r = _add_vars!(model, nvar, length(witnesses))
    loc_map = _make_loc_map(nloc, witnesses)

    for (i1, wit) in enumerate(witnesses)
        for posevid in wit.pos_evids
            _add_pos_constr_prob!(prob, model, lfs, r, i1, posevid)
        end
        for lieevid in wit.lie_evids
            for i2 in loc_map[lieevid.loc2]
                _add_lie_constr_prob!(prob, model, lfs, r, i1, i2, lieevid)
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

    return PolyFunc([_value(lf) for lf in lfs], loc_map), value(r)
end

## Feasibility

struct GeneratorFeasibility <: GeneratorProblem
    ϵ::Float64
    δ::Float64
end

function _add_pos_constr_prob!(
        prob::GeneratorFeasibility, model, lfs, r, i, posevid
    )
    point = posevid.point
    α = posevid.npoint
    β = posevid.npoint/prob.ϵ
    _add_pos_constr!(model, lfs, r, i, point, α, β)
end

function _add_lie_constr_prob!(
        prob::GeneratorFeasibility, model, lfs, r, i1, i2, lieevid
    )
    point1 = lieevid.point1
    point2 = lieevid.point2
    α = lieevid.npoint1
    β = lieevid.npoint1*prob.δ
    _add_lie_constr(model, lfs, r, i1, i2, point1, point2, α, β)
end

function compute_polyf_feasibility(
        gen::Generator, ϵ::Float64, δ::Float64, solver
    )
    prob = GeneratorFeasibility(ϵ, δ)
    return _compute_polyf(prob, gen.nvar, gen.nloc, gen.witnesses, solver)
end

## Chebyshev

struct GeneratorChebyshev <: GeneratorProblem end

function _add_pos_constr_prob!(
        ::GeneratorChebyshev, model, lfs, r, i, posevid
    )
    point = posevid.point
    α = posevid.npoint
    _add_pos_constr!(model, lfs, r, i, point, α, 0)
end

function _add_lie_constr_prob!(
        prob::GeneratorChebyshev, model, lfs, r, i1, i2, lieevid
    )
    point1 = lieevid.point1
    point2 = lieevid.point2
    α::Float64 = i1 == i2 ? lieevid.ndiff : lieevid.npoint1 + lieevid.npoint2
    # α::Float64 = i1 == i2 ? lieevid.ndiff : 2*lieevid.npoint1 + lieevid.ndiff # old
    _add_lie_constr(model, lfs, r, i1, i2, point1, point2, α, 0)
end

function compute_polyf_chebyshev(gen::Generator, solver)
    prob = GeneratorChebyshev()
    return _compute_polyf(prob, gen.nvar, gen.nloc, gen.witnesses, solver)
end

## Witness

struct GeneratorWitness <: GeneratorProblem end

function _add_pos_constr_prob!(
        ::GeneratorWitness, model, lfs, r, i, posevid
    )
    point = posevid.point
    α = posevid.npoint
    _add_pos_constr!(model, lfs, r, i, point, α, 0)
end

function _add_lie_constr_prob!(
        prob::GeneratorWitness, model, lfs, r, i1, i2, lieevid
    )
    point1 = lieevid.point1
    point2 = lieevid.point2
    α::Float64 = i1 == i2 ?
        lieevid.npoint1*lieevid.nD : lieevid.npoint1*(1 + lieevid.nA)
    _add_lie_constr(model, lfs, r, i1, i2, point1, point2, α, 0)
end

function compute_polyf_witness(gen::Generator, solver)
    prob = GeneratorWitness()
    return _compute_polyf(prob, gen.nvar, gen.nloc, gen.witnesses, solver)
end