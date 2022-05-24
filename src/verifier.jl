module Verifier

using LinearAlgebra
using JuMP
const _RSC_ = JuMP.MathOptInterface.ResultStatusCode
const _TSC_ = JuMP.MathOptInterface.TerminationStatusCode
using ..Polyhedra: Cone

_VT_ = Vector{Float64}
_MT_ = Matrix{Float64}

struct VerifyingProblem
    nvar::Int
    domain::Cone
    A::_MT_
end

function _verify_pos_comp(nvar, domain, vecs, k, s, solver)
    model = Model(solver)
    x = @variable(model, [1:nvar], lower_bound=-1, upper_bound=1)
    fix(x[k], s, force=true)
    r = @variable(model, lower_bound=-2)

    for s in domain.supps
        @constraint(model, dot(s.a, x) ≤ 0)
    end

    for vec in vecs
        @constraint(model, r ≥ dot(vec, x))
    end

    @objective(model, Min, r)

    optimize!(model)

    if primal_status(model) == _RSC_(1) &&
            termination_status(model) == _TSC_(1)
        return value.(x), value(r)
    elseif primal_status(model) == _RSC_(0) &&
            termination_status(model) == _TSC_(2)
        return Float64[], Inf
    else
        error(string(
            "Verifier pos: neither feasible or infeasible: ",
            primal_status(model), " ",
            dual_status(model), " ",
            termination_status(model)
        ))
    end
end

function verify_pos(verif::VerifyingProblem, vecs::Vector{_VT_}, solver)
    xopt = Float64[]
    ropt = Inf
    for (k, s) in Iterators.product(1:verif.nvar, (-1, 1))
        x, r = _verify_pos_comp(verif.nvar, verif.domain, vecs, k, s, solver)
        if r < ropt
            ropt = r
            xopt = x
        end
    end
    if isinf(ropt)
        error(string("Verifier pos: infeasible: ", ropt))
    end
    return xopt, ropt
end

function verify_pos(verifs::Vector{VerifyingProblem}, vecs, solver)
    xopt = Float64[]
    ropt = Inf
    qopt = 0
    for (q, verif) in enumerate(verifs)
        x, r = verify_pos(verif, vecs, solver)
        if r < ropt
            ropt = r
            xopt = x
            qopt = q
        end
    end
    return xopt, ropt, qopt
end

function _verify_lie_comp(nvar, domain, A, vecs, k, s, i, solver)
    model = Model(solver)
    # new:
    x = @variable(model, [1:nvar], lower_bound=-1, upper_bound=1)
    fix(x[k], s, force=true) # end new
    # x = @variable(model, [1:nvar], lower_bound=-1e5, upper_bound=1e5) # old
    vec = vecs[i]

    for s in domain.supps
        @constraint(model, dot(s.a, x) ≤ 0)
    end

    # @constraint(model, dot(vec, x) == 1) # old

    for j = 1:length(vecs)
        j == i && continue
        vec2 = vecs[j]
        @constraint(model, dot(vec - vec2, x) ≥ 0)
    end

    @objective(model, Max, dot(vec, A, x))

    optimize!(model)

    if primal_status(model) == _RSC_(1) &&
            termination_status(model) == _TSC_(1)
        return value.(x), objective_value(model)
    elseif primal_status(model) == _RSC_(0) &&
            termination_status(model) == _TSC_(2)
        return Float64, -Inf
    else
        error(string(
            "Verifier lie: neither feasible or infeasible: ",
            primal_status(model), " ",
            dual_status(model), " ",
            termination_status(model)
        ))
    end

    return value.(x), objective_value(model)
end

function verify_lie(verif::VerifyingProblem, vecs, solver)
    xopt = Float64[]
    ropt = -Inf
    for (i, k, s) in Iterators.product(1:length(vecs), 1:verif.nvar, (-1, 1)) # new
    # for (i, k, s) in Iterators.product(1:length(vecs), 1:1, 1:1) # old
        x, r = _verify_lie_comp(
            verif.nvar, verif.domain, verif.A, vecs, k, s, i, solver
        )
        if r > ropt
            ropt = r
            xopt = x
        end
    end
    if isinf(ropt)
        error(string("Verifier lie: infeasible: ", ropt))
    end
    return xopt, ropt
end

function verify_lie(verifs::Vector{VerifyingProblem}, vecs, solver)
    xopt = Float64[]
    ropt = -Inf
    qopt = 0
    for (q, verif) in enumerate(verifs)
        x, r = verify_lie(verif, vecs, solver)
        # r = r/norm(x) # old
        if r > ropt
            ropt = r
            xopt = x
            qopt = q
        end
    end
    return xopt, ropt, qopt
end

end # module