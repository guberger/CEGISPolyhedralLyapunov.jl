struct VerifyingPos
    nvar::Int
    domain::Cone
end

struct VerifyingLie
    nvar::Int
    domain::Cone
    A::_MT_
end

struct Verifier
    pos_verifs::Vector{VerifyingPos}
    lie_verifs::Vector{VerifyingLie}
end

Verifier() = Verifier(VerifyingPos[], VerifyingLie[])

function add_verifying!(verif::Verifier, posverif::VerifyingPos)
    push!(verif.pos_verifs, posverif)
end

function add_verifying!(verif::Verifier, lieverif::VerifyingLie)
    push!(verif.lie_verifs, lieverif)
end

function add_verifying_pos!(verif::Verifier, nvar, domain)
    add_verifying!(verif, VerifyingPos(nvar, domain))
end

function add_verifying_lie!(verif::Verifier, nvar, domain, A)
    add_verifying!(verif, VerifyingLie(nvar, domain, A))
end

## Verif Pos

function _verify_pos_comp(nvar, domain, lfs, k, s, solver)
    model = Model(solver)
    x = @variable(model, [1:nvar], lower_bound=-1, upper_bound=1)
    fix(x[k], s, force=true)
    r = @variable(model, lower_bound=-2)

    for s in domain.supps
        @constraint(model, dot(s.a, x) ≤ 0)
    end

    for lf in lfs
        @constraint(model, r ≥ _eval(lf, x))
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

function _verify_pos_single(nvar, domain, lfs, solver)
    xopt::_VT_ = Float64[]
    ropt::Float64 = Inf
    for (k, s) in Iterators.product(1:nvar, (-1, 1))
        x, r = _verify_pos_comp(nvar, domain, lfs, k, s, solver)
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

function verify_pos(verif::Verifier, lfs::Vector{LinForm}, solver)
    xopt = Float64[]
    ropt = Inf
    qopt = 0
    for (q, posverif) in enumerate(verif.pos_verifs)
        x, r = _verify_pos_single(posverif.nvar, posverif.domain, lfs, solver)
        if r < ropt
            ropt = r
            xopt = x
            qopt = q
        end
    end
    return xopt, ropt, qopt
end

## Verify Lie

function _verify_lie_comp(nvar, domain, A, lfs, k, s, i, solver)
    model = Model(solver)
    # new:
    x = @variable(model, [1:nvar], lower_bound=-1, upper_bound=1)
    fix(x[k], s, force=true) # end new
    # x = @variable(model, [1:nvar], lower_bound=-1e5, upper_bound=1e5) # old
    lf = lfs[i]

    for s in domain.supps
        @constraint(model, dot(s.a, x) ≤ 0)
    end

    # @constraint(model, dot(lf, x) == 1) # old

    for j = 1:length(lfs)
        j == i && continue
        lf2 = lfs[j]
        @constraint(model, _eval(lf, x) ≥ _eval(lf2, x))
    end

    @objective(model, Max, dot(lf.lin, A, x))

    optimize!(model)

    if primal_status(model) == _RSC_(1) &&
            termination_status(model) == _TSC_(1)
        return value.(x), objective_value(model)
    elseif primal_status(model) == _RSC_(0) &&
            termination_status(model) == _TSC_(2)
        return Float64[], -Inf
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

function _verify_lie_single(nvar, domain, A, lfs, solver)
    xopt::_VT_ = Float64[]
    ropt::Float64 = -Inf
    for (i, k, s) in Iterators.product(1:length(lfs), 1:nvar, (-1, 1)) # new
    # for (i, k, s) in Iterators.product(1:length(lfs), 1:1, 1:1) # old
        x, r = _verify_lie_comp(nvar, domain, A, lfs, k, s, i, solver)
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

function verify_lie(verif::Verifier, lfs::Vector{LinForm}, solver)
    xopt = Float64[]
    ropt = -Inf
    qopt = 0
    for (q, lieverif) in enumerate(verif.lie_verifs)
        x, r = _verify_lie_single(
            lieverif.nvar, lieverif.domain, lieverif.A, lfs, solver
        )
        # r = r/norm(x) # old
        if r > ropt
            ropt = r
            xopt = x
            qopt = q
        end
    end
    return xopt, ropt, qopt
end