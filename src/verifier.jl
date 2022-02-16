function verify_PLF!(M, dim, x_opt, systems, coeffs, ζ, solver)
    obj_max = -Inf
    x_tmp = _VT_(undef, dim)
    i_opt = 0
    q_opt = 0
    σ_opt = 0
    flag = false
    flag_prob = false
    
    for i = 1:M
        flag_prob && break
        c = coeffs[i]

        for (q, sys) in enumerate(systems)
            model = Model(solver)
            x = @variable(model, [1:dim], lower_bound=-ζ, upper_bound=+ζ)

            @constraint(model, dot(c, x) == 1)

            for j = 1:M
                j == i && continue
                d = coeffs[j]
                @constraint(model, dot(d, x) ≤ 1)
            end

            @constraint(model, sys.domain*x .≤ 0)

            for (σ, A) in enumerate(sys.fields)
                @objective(model, Max, dot(c, A*x))

                optimize!(model)

                TS = Int(termination_status(model))
                PS = Int(primal_status(model))

                if isone(PS)
                    flag = true
                    map!(xv -> value(xv), x_tmp, x)
                    obj_val = objective_value(model)/norm(x_tmp)
                    if obj_val > obj_max
                        obj_max = obj_val
                        copyto!(x_opt, x_tmp)
                        i_opt, q_opt, σ_opt = i, q, σ
                    end
                elseif TS == 2 && iszero(PS)
                    # nothing
                else
                    println("Problem in verifying PLF")
                    @printf("status: %s, %s, %s\n", get_status(model)...)
                    @printf("i: %d, q: %d, σ: %d\n", i, q, σ)
                    flag_prob = true
                    break
                end
            end
        end
    end

    flag = !flag_prob && flag

    return obj_max, flag, i_opt, q_opt, σ_opt
end