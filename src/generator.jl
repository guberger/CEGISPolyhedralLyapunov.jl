struct PosEvidence
    loc::Int
    i::Int
    point::Point
    npoint::Float64
end

struct LieEvidence
    loc1::Int
    i1::Int
    point1::Point
    loc2::Int
    point2::Point
    npoint1::Float64
    npoint2::Float64
    ndiff::Float64
    nA::Float64
    nD::Float64
end

struct Generator
    nvar::Int
    nlfs::Vector{Int}
    pos_evids::Vector{PosEvidence}
    lie_evids::Vector{LieEvidence}
end

Generator(nvar::Int, nloc::Int) = Generator(
    nvar, zeros(Int, nloc), PosEvidence[], LieEvidence[]
)

function add_lf!(gen::Generator, loc)
    return gen.nlfs[loc] += 1
end

function add_evidence_pos!(gen::Generator, loc, i, point, npoint)
    push!(gen.pos_evids, PosEvidence(loc, i, point, npoint))
end

function add_evidence_lie!(
        gen::Generator,
        loc1, i1, point1, loc2, point2, npoint1, npoint2, ndiff, nA, nD
    )
    push!(gen.lie_evids, LieEvidence(
        loc1, i1, point1, loc2, point2, npoint1, npoint2, ndiff, nA, nD
    ))
end

## Compute lfs

struct _LF
    lin::Vector{VariableRef}
end
_eval(lf::_LF, point) = dot(point, lf.lin)

struct _PF
    lfs::Vector{_LF}
end

function _add_vars!(model, nvar, nlfs)
    pfs = Vector{_PF}(undef, length(nlfs))
    for (loc, nlf) in enumerate(nlfs)
        pfs[loc] = _PF(Vector{_LF}(undef, nlf))
        for i = 1:nlf
            lin = @variable(model, [1:nvar], lower_bound=-1, upper_bound=1)
            pfs[loc].lfs[i] = _LF(lin)
            #new:
            alin = @variable(model, [1:nvar], lower_bound=0, upper_bound=1)
            @constraint(model, -lin .≤ alin)
            @constraint(model, +lin .≤ alin)
            @constraint(model, sum(alin) ≤ 1) # end new
        end
    end
    r = @variable(model, upper_bound=2)
    return pfs, r
end

function _add_pos_constr!(model, lf, r, point, α, β)
    @constraint(model, _eval(lf, point) ≥ α*r + β)
end

function _add_lie_constr(model, lf1, lf2, r, point1, point2, α, β)
    @constraint(model,
        _eval(lf2, point2) + α*r + β ≤ _eval(lf1, point1)
    )
end

_value(lf::_LF) = LinForm(value.(lf.lin))

abstract type GeneratorProblem end

_compute_mpf(prob::GeneratorProblem, gen::Generator, solver) = _compute_mpf(
    prob, gen.nvar, gen.nlfs, gen.pos_evids, gen.lie_evids, solver
)

function _compute_mpf(
        prob::GeneratorProblem, nvar, nlfs,
        pos_evids, lie_evids, solver
    )
    model = Model(solver)
    pfs, r = _add_vars!(model, nvar, nlfs)

    for posevid in pos_evids
        lf = pfs[posevid.loc].lfs[posevid.i]
        _add_pos_constr_prob!(prob, model, lf, r, posevid)
    end

    for lieevid in lie_evids
        lf1 = pfs[lieevid.loc1].lfs[lieevid.i1]
        for lf2 in pfs[lieevid.loc2].lfs
            _add_lie_constr_prob!(prob, model, lf1, lf2, r, lieevid)
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

    return MultiPolyFunc([
        PolyFunc([_value(lf) for lf in polyf.lfs]) for polyf in pfs
    ]), value(r)
end

## Feasibility

struct GeneratorFeasibility <: GeneratorProblem
    ϵ::Float64
    δ::Float64
end

function _add_pos_constr_prob!(
        prob::GeneratorFeasibility, model, lf, r, posevid
    )
    point = posevid.point
    α = posevid.npoint
    β = posevid.npoint/prob.ϵ
    _add_pos_constr!(model, lf, r, point, α, β)
end

function _add_lie_constr_prob!(
        prob::GeneratorFeasibility, model, lf1, lf2, r, lieevid
    )
    point1 = lieevid.point1
    point2 = lieevid.point2
    α = lieevid.npoint1
    β = lieevid.npoint1*prob.δ
    _add_lie_constr(model, lf1, lf2, r, point1, point2, α, β)
end

function compute_mpf_feasibility(
        gen::Generator, ϵ::Float64, δ::Float64, solver
    )
    prob = GeneratorFeasibility(ϵ, δ)
    return _compute_mpf(prob, gen, solver)
end

## Chebyshev

struct GeneratorChebyshev <: GeneratorProblem end

function _add_pos_constr_prob!(
        ::GeneratorChebyshev, model, lf, r, posevid
    )
    point = posevid.point
    α = posevid.npoint
    _add_pos_constr!(model, lf, r, point, α, 0)
end

function _add_lie_constr_prob!(
        prob::GeneratorChebyshev, model, lf1, lf2, r, lieevid
    )
    point1 = lieevid.point1
    point2 = lieevid.point2
    α::Float64 = lf1 == lf2 ?
        lieevid.ndiff : lieevid.npoint1 + lieevid.npoint2
    # α::Float64 = i1 == i2 ?
        # lieevid.ndiff : 2*lieevid.npoint1 + lieevid.ndiff # old
    _add_lie_constr(model, lf1, lf2, r, point1, point2, α, 0)
end

function compute_mpf_chebyshev(gen::Generator, solver)
    prob = GeneratorChebyshev()
    return _compute_mpf(prob, gen, solver)
end

## Evidence

struct GeneratorEvidence <: GeneratorProblem end

function _add_pos_constr_prob!(
        ::GeneratorEvidence, model, lf, r, posevid
    )
    point = posevid.point
    α = posevid.npoint
    _add_pos_constr!(model, lf, r, point, α, 0)
end

function _add_lie_constr_prob!(
        prob::GeneratorEvidence, model, lf1, lf2, r, lieevid
    )
    point1 = lieevid.point1
    point2 = lieevid.point2
    α::Float64 = lf1 == lf2 ?
        lieevid.npoint1*lieevid.nD : lieevid.npoint1*(1 + lieevid.nA)
    _add_lie_constr(model, lf1, lf2, r, point1, point2, α, 0)
end

function compute_mpf_evidence(gen::Generator, solver)
    prob = GeneratorEvidence()
    return _compute_mpf(prob, gen, solver)
end