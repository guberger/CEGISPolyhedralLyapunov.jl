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

# function verify_candidate_lyapunov_function(method::VerifyPolyhedralSingle,
#                                             A_list, c_list, solver)
#     D = state_dim(method)
#     M = length(c_list)
#     Q = length(A_list)

#     model = Model(solver)
#     x_tab = [@variable(model, [1:D], base_name=string("x", i, "-", q),
#         lower_bound=-1.0, upper_bound=1.0) for i = 1:M, q = 1:Q]

#     for i = 1:M
#         c = c_list[i]
#         for j = 1:M
#             j == i && continue
#             d = c_list[j]
#             for q = 1:Q
#                 x = x_tab[i, q]
#                 @constraint(model, (c + d)'*x ≥ 0)
#                 @constraint(model, (c - d)'*x ≥ 0)
#             end
#         end
#     end

#     @objective(model, Max,
#         sum(c_list[i]'*(A_list[q]*x_tab[i, q]) for i = 1:M, q = 1:Q))
    
#     optimize!(model)

#     obj_max = -Inf
#     x_opt = zeros(D)
#     i_opt = 0
#     q_opt = 0
#     flag = false

#     if isone(Int(primal_status(model)))
#         flag = true
#         for i = 1:M, q = 1:Q
#             x = value.(x_tab[i, q])
#             obj_val = c_list[i]'*(A_list[q]*x)
#             if obj_val > obj_max
#                 obj_max = obj_val
#                 x_opt = x
#                 i_opt, q_opt = i, q
#             end
#         end
#     else
#         println("Problem in verifying Lyapunov function")
#         println(string.(termination_status(model),
#                         primal_status(model),
#                         dual_status(model)))
#     end

#     return obj_max, x_opt, flag, i_opt, q_opt
# end