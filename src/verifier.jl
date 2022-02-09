"""
    verify_candidate_lyapunov_function(method, sys,
                                       c_list, tol_faces, solver)

`AH_list` is a set of `(A_set, H_set)` where `A_set` is a set of matrices and
`H_set` is a set of vectors `c` defining a cone by the inequalities `c'*x ≤ 0`.
"""
function verify_candidate_lyapunov_function(method::VerifyPolyhedralMultiple,
                                            sys::PiecewiseLinearSystem,
                                            c_list, tol_faces, solver)
    D = state_dim(method)
    M = length(c_list)
    Q = sys.n_mode
    obj_max = -Inf
    x_opt = zeros(D)
    j_opt = 0
    q_opt = 0
    qA_opt = 0
    flag = false
    flag_prob = false
    x_bound = 2/max(tol_faces, 1e-9)
    
    for j = 1:M
        flag_prob && break
        c = c_list[j]

        for q = 1:Q
            A_set, H_set = sys.As_list[q], sys.Hs_list[q]
            model = Model(solver)
            x = @variable(model, [1:D], base_name="x",
                    lower_bound=-x_bound, upper_bound=x_bound)

            @constraint(model, c'*x == 1)

            for k = 1:M
                k == j && continue
                d = c_list[k]
                @constraint(model, d'*x ≤ 1)
            end

            for h in H_set
                @constraint(model, h'*x ≤ 0)
            end

            for (qA, A) in enumerate(A_set)
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
                        j_opt, q_opt, qA_opt = j, q, qA
                    end
                elseif TS == 2 && iszero(PS)
                    # nothing
                else
                    println("Problem in verifying Lyapunov function")
                    @printf("j: %d, q: %d, qA: %d\n", j, q, qA)
                    println(string.((termination_status(model),
                                    primal_status(model),
                                    dual_status(model))))
                    flag_prob = true
                    break
                end
            end
        end
    end

    flag = !flag_prob && flag

    return obj_max, x_opt, flag, j_opt, q_opt, qA_opt
end