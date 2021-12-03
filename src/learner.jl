function learn_candidate_lyapunov_function(
        method::PolyhedralPointwise, x_dx_list, Gain, solver)
    D = state_dim(method)
    N = length(x_dx_list)
    model = Model(solver)
    c_list = [@variable(model, [1:D], base_name=string("c", i),
        lower_bound=-1.0, upper_bound=1.0) for i = 1:N]
    r = @variable(model)
    C = Gain

    for i = 1:N
        xt = x_dx_list[i][1]
        nxt = norm(xt)
        x = xt/nxt
        c = c_list[i]
        for dxt in x_dx_list[i][2]
            dx = dxt/nxt
            ndx = norm(dx)
            @constraint(model, dx'*c + ndx*r ≤ 0)
            for j = 1:N
                j == i && continue
                d = c_list[j]
                @constraint(model, x'*(c + d) ≥ 0)
                @constraint(model, x'*(c - d) ≥ 0)
                @constraint(model, (+dx)'*d - C*x'*(c - d) + ndx*r ≤ 0)
                @constraint(model, (-dx)'*d - C*x'*(c + d) + ndx*r ≤ 0)
            end
        end
    end

    @objective(model, Max, r)

    optimize!(model)

    if has_values(model)
        ropt = value(r)
        copt_list = map(c -> value.(c), c_list)
    else
        ropt = -1.0
        copt_list = Vector{Vector{Float64}}(undef, N)
        for i = 1:N
            copt_list[i] = zeros(D)
        end
    end

    return ropt, copt_list,
        (termination=termination_status(model),
        primal=primal_status(model),
        dual=dual_status(model))
end