function _make_constraints_adaptive(model, coeffs, δ, M, G, ϵ, i, flow)
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

function _learn_PLF_adaptive(M, dim, flows, G, ϵ, solver)
    model = Model(solver)
    coeffs = [@variable(model, [1:dim], lower_bound=-1, upper_bound=+1)
              for i = 1:M]
    δ = @variable(model, lower_bound=0)

    for (i, flow) in enumerate(flows)
        _make_constraints_adaptive(model, coeffs, δ, M, G, ϵ, i, flow)
    end

    @objective(model, Max, δ)

    optimize!(model)

    if has_values(model)
        δ_opt = value(δ)
        coeffs_opt = map(c -> value.(c), coeffs)
    else
        δ_opt = -1.0
        coeffs_opt = [zeros(dim) for i = 1:M]
    end

    return δ_opt, coeffs_opt, get_status(model)
end

function learn_PLF_adaptive(dim, flows,
                            G0, Gmax, r0, rmin,
                            ϵ, solver; output=true)
    
    M = length(flows)
    G = G0
    r = r0

    if iszero(M)
        return Inf, _VT_[], G, r, true # δ, c_list, G, r, flag
    end

    δ = -1.0
    coeffs = [zeros(dim) for i = 1:M]
    status = (_TSC_(0), _RSC_(0), _RSC_(0))
    iter = 0
    flag = false

    while G ≤ Gmax && r ≥ rmin
        iter += 1
        if output
            @printf("iter: %d. G: %f, r: %f\n", iter, G, r)
        end
        δ, coeffs, status = _learn_PLF_adaptive(M, dim, flows, G, ϵ, solver)
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

    return δ, coeffs, G, r, flag
end

function _make_constraints_fixed(model, coeffs, δ, node)
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

function learn_PLF_fixed!(M, dim, coeffs_opt, nodes, solver; output=true)
    if isempty(nodes)
        return Inf, true # δ, flag
    end

    N = length(coeffs_opt)

    model = Model(solver)
    coeffs = [@variable(model, [1:dim], lower_bound=-1, upper_bound=+1)
              for i = 1:N]
    for i = M+1:N
        for k = 1:dim
            fix(coeffs[i][k], coeffs_opt[i][k], force=true)
        end
    end
    δ = @variable(model)

    for node in nodes
        _make_constraints_fixed(model, coeffs, δ, node)
    end

    @objective(model, Max, δ)

    optimize!(model)

    δ_opt = value(δ)
    for i = 1:M
        map!(cv -> value(cv), coeffs_opt[i], coeffs[i])
    end

    if output
        @printf("status: %s, %s, %s; δ: %f\n", get_status(model)..., δ_opt)
    end

    flag = primal_status(model) == _RSC_(1)

    return δ_opt, flag
end