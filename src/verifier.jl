function verify_candidate_lyapunov_function(method::VerifyPolyhedralMultiple,
                                            A_list, c_list, tol_faces, solver)
    D = state_dim(method)
    M = length(c_list)
    obj_max = -Inf
    x_opt = zeros(D)
    j_opt = 0
    q_opt = 0
    flag = false
    flag_prob = false
    x_bound = 2/max(tol_faces, 1e-9)
    
    for j = 1:M
        flag_prob && break
        c = c_list[j]
        model = Model(solver)
        x = @variable(model, [1:D], base_name="x",
                lower_bound=-x_bound, upper_bound=x_bound)

        for k = 1:M
            k == j && continue
            d = c_list[k]
            @constraint(model, c'*x == 1)
            @constraint(model, +d'*x ≤ 1)
            @constraint(model, -d'*x ≤ 1)
        end

        for (q, A) in enumerate(A_list)
            @objective(model, Max, c'*(A*x))

            optimize!(model)

            TS = Int(termination_status(model))
            PS = Int(primal_status(model))

            if isone(PS)
                flag = true
                obj_val = objective_value(model)
                if obj_val > obj_max
                    obj_max = obj_val
                    x_opt = value.(x)
                    j_opt, q_opt = j, q
                end
            elseif TS == 2 && iszero(PS)
                # nothing
            else
                println("Problem in verifying Lyapunov function")
                @printf("j: %d, q: %d\n", j, q)
                println(string.((termination_status(model),
                                 primal_status(model),
                                 dual_status(model))))
                flag_prob = true
                break
            end
        end
    end

    flag = !flag_prob && flag

    return obj_max, x_opt, flag, j_opt, q_opt
end