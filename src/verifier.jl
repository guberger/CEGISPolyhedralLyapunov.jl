struct PosPredicate
    nvar::Int
    domain::Cone
    loc::Int
end

struct LieDiscPredicate
    nvar::Int
    domain::Cone
    loc1::Int
    A::_MT_
    loc2::Int
end

struct LieContPredicate
    nvar::Int
    domain::Cone
    loc::Int
    A::_MT_
end

struct Verifier
    pos_predics::Vector{PosPredicate}
    liedisc_predics::Vector{LieDiscPredicate}
    liecont_predics::Vector{LieContPredicate}
end

Verifier() = Verifier(PosPredicate[], LieDiscPredicate[], LieContPredicate[])

function add_predicate!(verif::Verifier, predic::PosPredicate)
    push!(verif.pos_predics, predic)
end

function add_predicate!(verif::Verifier, predic::LieDiscPredicate)
    push!(verif.liedisc_predics, predic)
end

function add_predicate!(verif::Verifier, predic::LieContPredicate)
    push!(verif.liecont_predics, predic)
end

## Optim problem

abstract type VerifierProblem end

function _add_variables!(prob, model)
    x = @variable(model, [1:prob.nvar], lower_bound=-1, upper_bound=1)
    r = @variable(model, upper_bound=2)
    return x, r
end

function _add_domain_constrs!(prob, model, x)
    for s in prob.domain.supps
        @constraint(model, dot(s.a, x) ≤ 0)
    end
end

_feasib(model) =
    primal_status(model) == _RSC_(1) && termination_status(model) == _TSC_(1)
_infeas(model) =
    primal_status(model) == _RSC_(0) && termination_status(model) == _TSC_(2)

function _optimize!(model)
    optimize!(model)
    if _feasib(model)
        return objective_value(model), true
    elseif _infeas(model)
        return -Inf, false
    else
        error(string(
            "Verifier: neither feasible or infeasible: ",
            primal_status(model), " ",
            dual_status(model), " ",
            termination_status(model)
        ))
    end
end

function _verify!(prob::VerifierProblem, pfs, xrec, solver)
    model = solver()
    x, r = _add_variables!(prob, model)

    _add_domain_constrs!(prob, model, x)
    _add_constrs_prob!(prob, model, x, r, pfs)

    @objective(model, Max, r)

    ropt = -Inf
    flag_feas = false
    
    for k in eachindex(x)
        lb = lower_bound(x[k])
        ub = upper_bound(x[k])
        for s in (-1, 1)
            fix(x[k], s, force=true)
            obj, F = _optimize!(model)
            flag_feas |= F
            if F && (obj > ropt)
                resize!(xrec, length(x))
                map!(xk -> value(xk), xrec, x)
                ropt = obj
            end
        end
        unfix(x[k])
        set_lower_bound(x[k], lb)
        set_upper_bound(x[k], ub)
    end

    return ropt, flag_feas
end

## Verif Pos
struct VerifierPos <: VerifierProblem
    nvar::Int
    domain::Cone
    loc::Int
end

function _add_constrs_prob!(prob::VerifierPos, model, x, r, pfs)
    for lf in pfs[prob.loc].lfs
        @constraint(model, 0 ≥ _eval(lf, x) + r)
    end
end

function verify_pos(verif::Verifier, mpf::MultiPolyFunc, solver)
    xrec = Float64[]
    xopt = Float64[]
    ropt::Float64 = -Inf
    locopt::Int = 0
    for predic in verif.pos_predics
        prob = VerifierPos(predic.nvar, predic.domain, predic.loc)
        rc, flag_feas = _verify!(prob, mpf.pfs, xrec, solver)
        @assert flag_feas
        if rc > ropt
            resize!(xopt, length(xrec))
            copyto!(xopt, xrec)
            ropt = rc
            locopt = predic.loc
        end
    end
    @assert !isinf(ropt)
    return xopt, ropt, locopt
end

## Verify Lie Disc
struct VerifierLieDisc <: VerifierProblem
    nvar::Int
    domain::Cone
    loc1::Int
    A::_MT_
    loc2::Int
    i2::Int
end

function _add_constrs_prob!(prob::VerifierLieDisc, model, x, r, pfs)
    val2 = _eval(pfs[prob.loc2].lfs[prob.i2], prob.A*x)
    for lf1 in pfs[prob.loc1].lfs
        @constraint(model, val2 ≥ _eval(lf1, x) + r)
    end
end

function verify_lie_disc(verif::Verifier, mpf::MultiPolyFunc, solver)
    xrec = Float64[]
    xopt = Float64[]
    ropt::Float64 = -Inf
    locopt::Int = 0
    for predic in verif.liedisc_predics
        for i2 in eachindex(mpf.pfs[predic.loc2].lfs)
            prob = VerifierLieDisc(
                predic.nvar, predic.domain,
                predic.loc1, predic.A, predic.loc2, i2
            )
            rc, flag_feas = _verify!(prob, mpf.pfs, xrec, solver)
            @assert flag_feas
            if rc > ropt
                resize!(xopt, length(xrec))
                copyto!(xopt, xrec)
                ropt = rc
                locopt = predic.loc1
            end
        end
    end
    @assert isempty(verif.liedisc_predics) || !isinf(ropt)
    return xopt, ropt, locopt
end

## Verify Lie Cont
struct VerifierLieCont <: VerifierProblem
    nvar::Int
    domain::Cone
    loc::Int
    i::Int
    A::_MT_
end

function _add_constrs_prob!(prob::VerifierLieCont, model, x, r, pfs)
    lf1 = pfs[prob.loc].lfs[prob.i]
    for lf2 in pfs[prob.loc].lfs
        lf1 == lf2 && continue
        @constraint(model, _eval(lf1, x) ≥ _eval(lf2, x))
    end
    @constraint(model, _eval(lf1, prob.A*x) ≥ r)
end

function verify_lie_cont(verif::Verifier, mpf::MultiPolyFunc, solver)
    xrec = Float64[]
    xopt = Float64[]
    ropt::Float64 = -Inf
    locopt::Int = 0
    for predic in verif.liecont_predics
        flag_feas = false
        for i in eachindex(mpf.pfs[predic.loc].lfs)
            prob = VerifierLieCont(
                predic.nvar, predic.domain, predic.loc, i, predic.A
            )
            rc, F = _verify!(prob, mpf.pfs, xrec, solver)
            if F && (rc > ropt)
                resize!(xopt, length(xrec))
                copyto!(xopt, xrec)
                ropt = rc
                locopt = predic.loc
            end
            flag_feas |= F
        end
        @assert flag_feas
    end
    @assert isempty(verif.liecont_predics) || !isinf(ropt)
    return xopt, ropt, locopt
end