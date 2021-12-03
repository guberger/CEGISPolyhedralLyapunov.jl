function verify_candidate_lyapunov_function(
        method::Polyhedral, A_list, c_list, solver)
    D = state_dim(method)
    M = length(c_list)
    Q = length(A_list)

    output_T = Tuple{Float64,Vector{Float64},OPT_status}
    output_list = Vector{output_T}(undef, M*Q)

    for (iter, elem) in enumerate(Iterators.product(1:M, 1:Q))
        i, q = elem
        c = c_list[i]
        A = A_list[q]

        model = Model(solver)
        x = @variable(model, [1:D], lower_bound=-1.0, upper_bound=1.0)

        for j = 1:M
            j == i && continue
            d = c_list[j]
            @constraint(model, (c + d)'*x ≥ 0)
            @constraint(model, (c - d)'*x ≥ 0)
        end

        @objective(model, Max, c'*(A*x))

        optimize!(model)

        if has_values(model)
            obj = objective_value(model)
            xopt = value.(x)
        else
            obj = Inf
            xopt = zeros(D)
        end

        status = (termination=termination_status(model),
            primal=primal_status(model),
            dual=dual_status(model))
        output_list[iter] = (obj, xopt, status)
    end

    return output_list
end