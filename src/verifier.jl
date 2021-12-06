function verify_candidate_lyapunov_function(method::VerifyPolyhedralHRep,
                                            A_list, c_list, solver)
    D = state_dim(method)
    M = length(c_list)
    Q = length(A_list)
    obj_max = -Inf
    x_opt = zeros(D)
    i_opt = 0
    q_opt = 0

    for i = 1:M
        c = c_list[i]
        model = Model(solver)
        x = @variable(model, [1:D], base_name="x",
            lower_bound=-1.0, upper_bound=1.0)

        for j = 1:M
            j == i && continue
            d = c_list[j]
            @constraint(model, (c + d)'*x ≥ 0)
            @constraint(model, (c - d)'*x ≥ 0)
        end

        for (q, A) in enumerate(A_list)
            @objective(model, Max, c'*(A*x))

            optimize!(model)

            if string(primal_status(model)) == "FEASIBLE_POINT"
                obj_val = objective_value(model)
                if obj_val > obj_max
                    obj_max = obj_val
                    x_opt = value.(x)
                    i_opt, q_opt = i, q
                end
            end
        end
    end

    return obj_max, x_opt, i_opt, q_opt
end