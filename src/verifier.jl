struct PosPredicate
    nvar::Int
    domain::Cone
    loc::Int
end

struct LiePredicate
    nvar::Int
    domain::Cone
    loc1::Int
    A::_MT_
    loc2::Int
end

struct Verifier
    pos_predics::Vector{PosPredicate}
    lie_predics::Vector{LiePredicate}
end

Verifier() = Verifier(PosPredicate[], LiePredicate[])

function add_predicate!(verif::Verifier, pospredic::PosPredicate)
    push!(verif.pos_predics, pospredic)
end

function add_predicate!(verif::Verifier, liepredic::LiePredicate)
    push!(verif.lie_predics, liepredic)
end

function add_predicate_pos!(verif::Verifier, nvar, domain, loc)
    add_predicate!(verif, PosPredicate(nvar, domain, loc))
end

function add_predicate_lie!(verif::Verifier, nvar, domain, loc1, A, loc2)
    add_predicate!(verif, LiePredicate(nvar, domain, loc1, A, loc2))
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

function _optim_update!(model, x, xopt, ropt)
    optimize!(model)
    if _feasib(model)
        obj = objective_value(model)
        if obj > ropt
            resize!(xopt, length(x))
            map!(xk -> value(xk), xopt, x)
            return obj, true
        else
            return ropt, true
        end
    elseif !_infeas(model)
        error(string(
            "Verifier: neither feasible or infeasible: ",
            primal_status(model), " ",
            dual_status(model), " ",
            termination_status(model)
        ))
    end
    return ropt, false
end

function _verify!(prob::VerifierProblem, lfs, xopt, ropt, solver)
    model = Model(solver)
    x, r = _add_variables!(prob, model)

    _add_domain_constrs!(prob, model, x)
    _add_pos_constrs!(prob, model, x, r, lfs)
    _add_lie_constrs!(prob, model, x, r, lfs)

    @objective(model, Max, r)

    flag_feas = false
    
    for k in eachindex(x)
        lb = lower_bound(x[k])
        ub = upper_bound(x[k])
        for s in (-1, 1)
            fix(x[k], s, force=true)
            ropt, F = _optim_update!(model, x, xopt, ropt)
            flag_feas |= F
        end
        unfix(x[k])
        set_lower_bound(x[k], lb)
        set_upper_bound(x[k], ub)
    end

    if !flag_feas
        error(string("Verifier: infeasible: ", ropt))
    end

    return ropt
end

## Verif Pos

struct VerifierPos <: VerifierProblem
    nvar::Int
    domain::Cone
    iset::BitSet
end

function _add_pos_constrs!(prob::VerifierPos, model, x, r, lfs)
    for i in prob.iset
        @constraint(model, 0 ≥ _eval(lfs[i], x) + r)
    end
end

_add_lie_constrs!(::VerifierPos, ::Any, ::Any, ::Any, ::Any) = nothing

function verify_pos(verif::Verifier, polyf::PolyFunc, solver)
    xopt = Float64[]
    ropt::Float64 = -Inf
    for p in verif.pos_predics
        prob = VerifierPos(p.nvar, p.domain, polyf.loc_map[p.loc])
        ropt = _verify!(prob, polyf.lfs, xopt, ropt, solver)
    end
    return xopt, ropt
end

## Verify Lie

struct VerifierLie <: VerifierProblem
    nvar::Int
    domain::Cone
    iset1::BitSet
    A::_MT_
    i2::Int
end

_add_pos_constrs!(::VerifierLie, ::Any, ::Any, ::Any, ::Any) = nothing

function _add_lie_constrs!(prob::VerifierLie, model, x, r, lfs)
    val2 = _eval(lfs[prob.i2], prob.A*x)
    for i1 in prob.iset1
        @constraint(model, val2 ≥ _eval(lfs[i1], x) + r)
    end
end

function verify_lie(verif::Verifier, polyf::PolyFunc, solver)
    xopt = Float64[]
    ropt::Float64 = -Inf
    for p in verif.lie_predics
        iset1 = polyf.loc_map[p.loc1]
        for i2 in polyf.loc_map[p.loc2]
            prob = VerifierLie(p.nvar, p.domain, iset1, p.A, i2)
            ropt = _verify!(prob, polyf.lfs, xopt, ropt, solver)
        end
    end
    return xopt, ropt
end