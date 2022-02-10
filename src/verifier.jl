function _make_xvars_verifier(model, C, dim)
    return @variable(model, [1:dim], lower_bound=-C, upper_bound=+C)
end

function _make_consts_verifier(model, x, coeffs, c, H)
    @constraint(model, dot(c, x) == 1)
    for d in coeffs
        d == c && continue
        @constraint(model, dot(d, x) ≤ 1)
    end
    @constraint(model, H*x .≤ 0)
    return nothing
end

function verify_PLF(dim, systems, coeffs, ϵ, solver)
    Q = length(systems)
    obj_max = -Inf
    x_opt = zeros(dim)
    i_opt = 0
    q_opt = 0
    σ_opt = 0
    flag = false
    flag_prob = false
    C = 2/max(ϵ, 1e-9)
    
    for (i, c) in enumerate(coeffs)
        flag_prob && break
        for (q, sys) in enumerate(systems)
            model = Model(solver)
            x = _make_xvars_verifier(model, C, dim)

            _make_consts_verifier(model, x, coeffs, c, sys.domain)

            for (σ, A) in enumerate(sys.fields)
                @objective(model, Max, dot(c, A*x))

                optimize!(model)

                TS = Int(termination_status(model))
                PS = Int(primal_status(model))

                if isone(PS)
                    flag = true
                    obj_val = objective_value(model)
                    if obj_val > obj_max
                        obj_max = obj_val
                        x_opt = value.(x)
                        i_opt, q_opt, σ_opt = i, q, σ
                    end
                elseif TS == 2 && iszero(PS)
                    # nothing
                else
                    println("Problem in verifying PLF")
                    @printf("j: %d, q: %d, σ: %d\n", j, q, σ)
                    println(string.(get_status(model)))
                    flag_prob = true
                    break
                end
            end
        end
    end

    flag = !flag_prob && flag

    return obj_max, x_opt, flag, i_opt, q_opt, σ_opt
end