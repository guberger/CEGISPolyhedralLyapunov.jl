function verify_candidate_lyapunov_function(method::VerifyPolyhedralMultiple,
                                            A_list, c_list, tol_faces, solver)
    D = state_dim(method)
    M = length(c_list)
    obj_max = -Inf
    x_opt = zeros(D)
    i_opt = 0
    q_opt = 0
    flag = false
    flag_prob = false
    x_bound = 2/max(tol_faces, 1e-9)

    for i = 1:M
        flag_prob && break
        c = c_list[i]
        model = Model(solver)
        x = @variable(model, [1:D], base_name="x",
                lower_bound=-x_bound, upper_bound=x_bound)

        for j = 1:M
            j == i && continue
            d = c_list[j]
            @constraint(model, c'*x == 1)
            @constraint(model, +d'*x ≤ 1)
            @constraint(model, -d'*x ≤ 1)
        end

        for (q, A) in enumerate(A_list)
            @objective(model, Max, c'*(A*x))

            optimize!(model)

            if isone(Int(primal_status(model)))
                flag = true
                obj_val = objective_value(model)
                if obj_val > obj_max
                    obj_max = obj_val
                    x_opt = value.(x)
                    i_opt, q_opt = i, q
                end
            else
                println("Problem in verifying Lyapunov function")
                @printf("i: %d, q: %d\n", i, d)
                println(string.(termination_status(model),
                                primal_status(model),
                                dual_status(model)))
                flag_prob = true
                break
            end
        end
    end

    flag = !flag_prob && flag

    return obj_max, x_opt, flag, i_opt, q_opt
end