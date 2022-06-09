struct PosEvidence
    loc::Int
    i::Int
    point::Point
    npoint::Float64
end

struct LieDiscEvidence
    loc1::Int
    i1::Int
    point1::Point
    loc2::Int
    point2::Point # A*point1
    npoint1::Float64
    npoint2::Float64
    ndiff::Float64 # norm(point2 - point1)
    nA::Float64 # opnorm(A)
    nD::Float64 # opnorm(A - I)
end

struct LieContEvidence
    loc::Int
    i::Int
    point1::Point
    point2::Point # point1*(I + τ*A)
    npoint1::Float64
    npoint2::Float64
    ndiff::Float64 # norm(A*point1)*τ
    nA::Float64 # opnorm(I + A*τ)
    nD::Float64 # opnorm(A*τ)
    τ::Float64
end

struct Generator
    nvar::Int
    nlfs::Vector{Int}
    pos_evids::Vector{PosEvidence}
    liedisc_evids::Vector{LieDiscEvidence}
    liecont_evids::Vector{LieContEvidence}
end

Generator(nvar::Int, nloc::Int) = Generator(
    nvar, zeros(Int, nloc),
    PosEvidence[], LieDiscEvidence[], LieContEvidence[]
)

function add_lf!(gen::Generator, loc)
    return gen.nlfs[loc] += 1
end

function add_evidence!(gen::Generator, evid::PosEvidence)
    push!(gen.pos_evids, evid)
end

function add_evidence!(gen::Generator, evid::LieDiscEvidence)
    push!(gen.liedisc_evids, evid)
end

function add_evidence!(gen::Generator, evid::LieContEvidence)
    push!(gen.liecont_evids, evid)
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
    prob, gen.nvar, gen.nlfs,
    gen.pos_evids, gen.liedisc_evids, gen.liecont_evids,
    solver
)

function _compute_mpf(
        prob::GeneratorProblem, nvar, nlfs,
        pos_evids, liedisc_evids, liecont_evids,
        solver
    )
    model = solver()
    pfs, r = _add_vars!(model, nvar, nlfs)

    for evid in pos_evids
        lf = pfs[evid.loc].lfs[evid.i]
        _add_constr_prob!(prob, model, lf, r, evid)
    end

    for evid in liedisc_evids
        lf1 = pfs[evid.loc1].lfs[evid.i1]
        for lf2 in pfs[evid.loc2].lfs
            _add_constr_prob!(prob, model, lf1, lf2, r, evid)
        end
    end

    for evid in liecont_evids
        lf1 = pfs[evid.loc].lfs[evid.i]
        for lf2 in pfs[evid.loc].lfs
            _add_constr_prob!(prob, model, lf1, lf2, r, evid)
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

function _add_constr_prob!(
        prob::GeneratorFeasibility, model, lf, r, evid::PosEvidence
    )
    point = evid.point
    α = evid.npoint
    β = evid.npoint/prob.ϵ
    _add_pos_constr!(model, lf, r, point, α, β)
end

function _add_constr_prob!(
        prob::GeneratorFeasibility, model, lf1, lf2, r, evid::LieDiscEvidence
    )
    point1 = evid.point1
    point2 = evid.point2
    α = evid.npoint1
    β = evid.npoint1*prob.δ
    _add_lie_constr(model, lf1, lf2, r, point1, point2, α, β)
end

function _add_constr_prob!(
        prob::GeneratorFeasibility, model, lf1, lf2, r, evid::LieContEvidence
    )
    point1 = evid.point1
    point2 = evid.point2
    α = evid.npoint1
    β = evid.npoint1*evid.τ*prob.δ
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

function _add_constr_prob!(
        ::GeneratorChebyshev, model, lf, r, evid::PosEvidence
    )
    point = evid.point
    α = evid.npoint
    _add_pos_constr!(model, lf, r, point, α, 0)
end

function _add_constr_prob!(
        ::GeneratorChebyshev, model, lf1, lf2, r,
        evid::Union{LieDiscEvidence,LieContEvidence}
    )
    point1 = evid.point1
    point2 = evid.point2
    α::Float64 = lf1 == lf2 ?
        evid.ndiff : evid.npoint1 + evid.npoint2
    # α::Float64 = i1 == i2 ?
    #     evid.ndiff : 2*evid.npoint1 + evid.ndiff # old
    _add_lie_constr(model, lf1, lf2, r, point1, point2, α, 0)
end

function compute_mpf_chebyshev(gen::Generator, solver)
    prob = GeneratorChebyshev()
    return _compute_mpf(prob, gen, solver)
end

## Evidence

struct GeneratorEvidence <: GeneratorProblem end

function _add_constr_prob!(
        ::GeneratorEvidence, model, lf, r, evid::PosEvidence
    )
    point = evid.point
    α = evid.npoint
    _add_pos_constr!(model, lf, r, point, α, 0)
end

function _add_constr_prob!(
        ::GeneratorEvidence, model, lf1, lf2, r,
        evid::Union{LieDiscEvidence,LieContEvidence}
    )
    point1 = evid.point1
    point2 = evid.point2
    α = evid.npoint1*(1 + evid.nA)
    α::Float64 = lf1 == lf2 ?
        evid.npoint1*evid.nD : evid.npoint1*(1 + evid.nA)
    _add_lie_constr(model, lf1, lf2, r, point1, point2, α, 0)
end

function compute_mpf_evidence(gen::Generator, solver)
    prob = GeneratorEvidence()
    return _compute_mpf(prob, gen, solver)
end