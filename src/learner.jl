function _make_cvars_learner(model, M, ::Val{D}) where D
    _cvar(i, k) = @variable(model, lower_bound=-1, upper_bound=1)
    return map(i -> SVector{D}(ntuple(k -> _cvar(i, k), Val(D))), 1:M)
end

function _make_consts_learner(model, coeffs, δ, witness, indexes, G, ϵ)
    xt = witness.point
    nxt = norm(xt)
    x = xt/nxt
    for i in indexes
        c = coeffs[i]
        if ϵ > 0
            @constraint(model, dot(x, c) ≥ ϵ)
        end
        for d in coeffs
            d == c && continue
            @constraint(model, dot(x, c - d) ≥ 0)
        end
        for dxt in witness.flows
            dx = dxt/nxt
            ndx = norm(dx)
            @constraint(model, dot(dx, c) + ndx*δ ≤ 0)
            for d in coeffs
                d == c && continue
                @constraint(model, dot(dx, d) - G*dot(x, c - d) + ndx*δ ≤ 0)
            end
        end
    end
    return nothing
end

function learn_PLF(M, witnesses, get_indexes,
                   G, ϵ, solver)
    D = state_dim(eltype(witnesses))
    model = Model(solver)
    coeffs = _make_cvars_learner(model, M, Val(D))
    δ = @variable(model, lower_bound=0)

    for witness in witnesses
        indexes = get_indexes(witness)
        _make_consts_learner(model, coeffs, δ, witness, indexes, G, ϵ)
    end

    @objective(model, Max, δ)

    optimize!(model)

    if has_values(model)
        δ_opt = value(δ)
        coeffs_opt = map(c -> value.(c), coeffs)
    else
        δ_opt = -1.0
        coeffs_opt = [zeros(SVector{D}) for i = 1:M]
    end

    return δ_opt, coeffs_opt, get_status(model)
end

function learn_PLF_params(M, witnesses, get_indexes,
                          G0, Gmax, r0, rmin,
                          ϵ, solver; output=true)
    D = state_dim(eltype(witnesses))
    G = G0
    r = r0

    if iszero(M)
        return Inf, SVector{D,Float64}[], G, r, true # δ, c_list, G, r, flag
    end

    δ = -1.0
    coeffs = [zeros(SVector{D}) for i = 1:M]
    s_status = ("Not started", "Unknown", "Unknown")
    iter = 0
    flag = false

    while G ≤ Gmax && r ≥ rmin
        iter += 1
        if output
            @printf("iter: %d. G: %f, r: %f\n", iter, G, r)
        end
        δ, coeffs, status = learn_PLF(M, witnesses, get_indexes,
                                      G, ϵ, solver)
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
