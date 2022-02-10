function _make_cvars_learner(model, M, dim)
    _cvar() = @variable(model, [1:dim], lower_bound=-1, upper_bound=+1)
    return map(i -> _cvar(), 1:M)
end

function _learn_PLF_robust(M, dim, flows, G, ϵ, solver)
    model = Model(solver)
    coeffs = _make_cvars_learner(model, M, dim)
    δ = @variable(model, lower_bound=0)

    for (i, flow) in enumerate(flows)
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

function learn_PLF_robust(dim, flows,
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
    s_status = ("Not started", "Unknown", "Unknown")
    iter = 0
    flag = false

    while G ≤ Gmax && r ≥ rmin
        iter += 1
        if output
            @printf("iter: %d. G: %f, r: %f\n", iter, G, r)
        end
        δ, coeffs, status = _learn_PLF_robust(M, dim, flows, G, ϵ, solver)
        s_status = string.(status)
        if output
            @printf("\tstatus: %s. δ: %f\n", s_status, δ)
        end
        flag = isone(Int(status[2])) && δ ≥ r
        (flag || 2*G > Gmax || r/2 < rmin) && break
        G = 2*G
        r = r/2
    end

    if !flag
        println("Problem in learning PLF")
        @printf("iter: %d. G: %f, r: %f\n", iter, G, r)
        println(s_status)
        println(δ)
    end

    return δ, coeffs, G, r, flag
end

function learn_PLF_branch(M, dim, nodes, ϵ, solver; output=true)
    model = Model(solver)
    coeffs = _make_cvars_learner(model, M, dim)
    δ = @variable(model, lower_bound=0)

    for node in nodes
        x = node.witness.flow.point
        i = node.witness.index
        c = coeffs[i]
        j = node.index
        d = coeffs[j]
        if ϵ > 0
            @constraint(model, dot(x, d) ≥ ϵ)
        end
        if i != j
            @constraint(model, dot(x, c - d) + 2*norm(x)*δ ≤ 0)
        elseif i == j
            for dx in node.witness.flow.grads
                @constraint(model, dot(dx, c) + norm(dx)*δ ≤ 0)
            end
        end
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

    if output
        @printf("status: %s. δ: %f\n", string.(get_status(model)), δ_opt)
    end

    flag = isone(Int(primal_status(model)))

    return δ_opt, coeffs_opt, flag
end