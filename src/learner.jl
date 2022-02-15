function _make_constraints_adaptive(model, M, coeffs, δ, G, ϵ, i, flow)
    xt = flow.point
    nxt = norm(xt)
    x = xt/nxt
    c = coeffs[i]
    if ϵ > 0
        @constraint(model, dot(x, c) ≥ ϵ)
    end
    for j = 1:M
        j == i && continue
        d = coeffs[j]
        @constraint(model, dot(x, c - d) ≥ 0)
    end
    for dxt in flow.grads
        dx = dxt/nxt
        ndx = norm(dx)
        @constraint(model, dot(dx, c) + ndx*δ ≤ 0)
        for j = 1:M
            j == i && continue
            d = coeffs[j]
            @constraint(model, dot(dx, d) - G*dot(x, c - d) + ndx*δ ≤ 0)
        end
    end
    return nothing
end

function _learn_PLF_adaptive!(Deb, M, dim, coeffs, coeffs_opt,
                              flows, G, ϵ, solver)
    model = Model(solver)
    for i = 1:M
        coeffs[i] = @variable(model, [1:dim], lower_bound=-1, upper_bound=+1)
    end
    δ = @variable(model, lower_bound=0)

    for (i, flow) in enumerate(flows)
        _make_constraints_adaptive(model, M, coeffs, δ, G, ϵ, i, flow)
    end

    @objective(model, Max, δ)

    optimize!(model)

    if has_values(model)
        δ_opt = value(δ)
        for i = 1:M
            map!(c -> value(c), coeffs_opt[Deb + i], coeffs[i])
        end
    else
        δ_opt = -1.0
    end

    return δ_opt, get_status(model)
end

function learn_PLF_adaptive!(Deb, M, dim, coeffs, coeffs_opt,
                             flows, G0, Gmax, r0, rmin,
                             ϵ, solver; output=true)
    G = G0
    r = r0

    if iszero(M)
        return Inf, G, r, true # δ, G, r, flag
    end

    δ = -1.0
    status = (_TSC_(0), _RSC_(0), _RSC_(0))
    iter = 0
    flag = false

    while G ≤ Gmax && r ≥ rmin
        iter += 1
        if output
            @printf("iter: %d. G: %f, r: %f\n", iter, G, r)
        end
        δ, status = _learn_PLF_adaptive!(Deb, M, dim, coeffs, coeffs_opt,
                                         flows, G, ϵ, solver)
        if output
            @printf("|---- status: %s, %s, %s; δ: %f\n", status..., δ)
        end
        flag = status[2] == _RSC_(1) && δ ≥ r
        (flag || 2*G > Gmax || r/2 < rmin) && break
        G = 2*G
        r = r/2
    end

    if !flag
        println("Problem in learning PLF")
        @printf("|--- iter: %d. G: %f, r: %f\n", iter, G, r)
        @printf("|--- status: %s, %s, %s; δ: %f\n", status..., δ)
    end

    return δ, G, r, flag
end

function _make_constraints_fixed_chebyshev(model, coeffs, δ, node)
    x = node.witness.flow.point
    i = node.witness.index
    c = coeffs[i]
    j = node.index
    d = coeffs[j]
    if i != j
        @constraint(model, dot(x, c - d) + 2*norm(x)*δ ≤ 0)
    elseif i == j
        for dx in node.witness.flow.grads
            @constraint(model, dot(dx, c) + norm(dx)*δ ≤ 0)
        end
    end
    return nothing
end

function learn_PLF_fixed!(::Chebyshev,
                          Deb, M, dim, coeffs, coeffs_opt,
                          nodes, solver; output=true)
    if isempty(nodes)
        return Inf, true # δ, flag
    end

    model = Model(solver)
    for i = Deb+1:Deb+M
        coeffs[i] = @variable(model, [1:dim], lower_bound=-1, upper_bound=+1)
    end
    δ = @variable(model)

    for node in nodes
        _make_constraints_fixed_chebyshev(model, coeffs, δ, node)
    end

    @objective(model, Max, δ)

    optimize!(model)

    δ_opt = value(δ)
    for i = Deb+1:Deb+M
        map!(cv -> value(cv), coeffs_opt[i], coeffs[i])
    end

    if output
        @printf("status: %s, %s, %s; δ: %f\n", get_status(model)..., δ_opt)
    end

    flag = primal_status(model) == _RSC_(1)

    return δ_opt, flag
end

function learn_PLF_fixed!(::MVE,
                          Deb, M, dim, coeffs, Δs, coeffs_opt,
                          nodes, solver; output=true)
    if isempty(nodes)
        return Inf, true # δ, flag
    end

    model = Model(solver)
    for i = Deb+1:Deb+M
        coeffs[i] = @variable(model, [1:dim], lower_bound=-1, upper_bound=+1)
    end
    δ = @variable(model)

    for node in nodes
        _make_constraints_fixed_chebyshev(model, coeffs, δ, node)
    end

    @objective(model, Max, δ)

    optimize!(model)

    δ_opt = value(δ)
    for i = Deb+1:Deb+M
        map!(cv -> value(cv), coeffs_opt[i], coeffs[i])
    end

    if output
        @printf("status: %s, %s, %s; δ: %f\n", get_status(model)..., δ_opt)
    end

    flag = primal_status(model) == _RSC_(1)

    return δ_opt, flag
end