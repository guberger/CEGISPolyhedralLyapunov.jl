##------------------------------------------------------------------------------
## Adaptive

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

function _learn_PLF_adaptive!(Deb, M, dim, coeffs_opt,
                              flows, G, ϵ, solver)
    model = Model(solver)
    coeffs = [@variable(model, [1:dim], lower_bound=-1, upper_bound=+1)
              for i = 1:M]
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

function learn_PLF_adaptive!(Deb, M, dim, coeffs_opt,
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
        δ, status = _learn_PLF_adaptive!(Deb, M, dim, coeffs_opt,
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

##------------------------------------------------------------------------------
## Fixed
# Chebyshev

function _make_constraints_fixed_chebyshev(model, Deb, coeffs_opt,
                                           coeffs, δ, node)
    x = node.witness.flow.point
    i = node.witness.index
    j = node.index
    c = coeffs[j - Deb]
    if i != j
        if i ≤ Deb
            diff = c - coeffs_opt[i]
            α = 1.0
        else
            diff = c - coeffs[i - Deb]
            α = sqrt(2)
        end
        @constraint(model, dot(x, diff) - α*norm(x)*δ ≥ 0)
    elseif i == j
        for dx in node.witness.flow.grads
            @constraint(model, dot(dx, c) + norm(dx)*δ ≤ 0)
        end
    end
    return nothing
end

function learn_PLF_fixed!(::Chebyshev,
                          Deb, M, dim, coeffs_opt,
                          nodes, solver; output=true)
    if isempty(nodes)
        return Inf, true # δ, flag
    end

    model = Model(solver)
    coeffs = [@variable(model, [1:dim]) for i = 1:M]
    δ = @variable(model)

    for i = 1:M
        @constraint(model, coeffs[i] .≤ +1 - δ)
        @constraint(model, coeffs[i] .≥ -1 + δ)
    end

    for node in nodes
        _make_constraints_fixed_chebyshev(model, Deb, coeffs_opt,
                                          coeffs, δ, node)
    end

    @objective(model, Max, δ)

    optimize!(model)

    δ_opt = value(δ)
    for i = 1:M
        map!(cv -> value(cv), coeffs_opt[Deb + i], coeffs[i])
    end

    if output
        @printf("status: %s, %s, %s; δ: %f\n", get_status(model)..., δ_opt)
    end

    flag = primal_status(model) == _RSC_(1)

    return δ_opt, flag
end

# MVE

function _make_constraints_fixed_mve(model, Deb, dim, coeffs_opt,
                                     coeffs, Q, node)
    x = node.witness.flow.point
    i = node.witness.index
    j = node.index
    c = coeffs[j - Deb]
    if i != j
        q = view(Q, :, (j - Deb - 1)*dim+1:(j - Deb)*dim)*x
        if i ≤ Deb
            diff = c - coeffs_opt[i]
        else
            diff = c - coeffs[i - Deb]
            q = q - view(Q, :, (i - Deb - 1)*dim+1:(i - Deb)*dim)*x
        end
        @constraint(model, vcat(dot(x, diff), q) in SecondOrderCone())
    elseif i == j
        Qv = view(Q, :, (j - Deb - 1)*dim+1:(j - Deb)*dim)
        for dx in node.witness.flow.grads
            @constraint(model, vcat(-dot(dx, c), Qv*dx) in SecondOrderCone())
        end
    end
    return nothing
end

function learn_PLF_fixed!(::MVE,
                          Deb, M, dim, coeffs_opt,
                          nodes, solver; output=true)
    if isempty(nodes)
        return Inf, true # δ, flag
    end

    N = M*dim
    model = Model(solver)
    coeffs = [@variable(model, [1:dim]) for i = 1:M]
    δ = @variable(model)
    Q = @variable(model, [1:N,1:N], PSD)
    
    Qup = [Q[i, j] for j = 1:N for i = 1:j]
    @constraint(model, vcat(δ, Qup) in MOI.RootDetConeTriangle(N))

    for i = 1:M
        c = coeffs[i]
        for k = 1:dim
            q = view(Q, :, (i - 1)*dim + k)
            @constraint(model, vcat(1 - c[k], q) in SecondOrderCone())
            @constraint(model, vcat(1 + c[k], q) in SecondOrderCone())
        end
    end

    for node in nodes
        _make_constraints_fixed_mve(model, Deb, dim, coeffs_opt,
                                    coeffs, Q, node)
    end

    @objective(model, Max, δ)

    optimize!(model)

    if has_values(model)
        δ_opt = value(δ)
        for i = 1:M
            map!(cv -> value(cv), coeffs_opt[Deb + i], coeffs[i])
        end
    else
        δ_opt = -1.0
    end

    if output
        @printf("status: %s, %s, %s; δ: %f\n", get_status(model)..., δ_opt)
    end

    flag = primal_status(model) == _RSC_(1)

    return δ_opt, flag
end