function process_PLF(systems, witnesses_init,
                     G0, Gmax, r0, rmin, ϵ, tol,
                     solver; kwargs...)
    D = state_dim(eltype(systems))
    WT_ = Witness{D}
    CT_ = SVector{D,Float64}
    coeffs = CT_[]
    output_period = get(kwargs, :output_period, 1)
    learner_output = get(kwargs, :learner_output, true)
    iter_max = get(kwargs, :iter_max, -1)
    do_trace = get(kwargs, :trace, true)

    # Trace
    trace = (coeffs_list=Vector{CT_}[],
             witnesses_list=Vector{WT_}[],
             flags_learner=Bool[],
             counterexample_list=WT_[],
             flags_verifier=Bool[])

    iter = 0
    G = G0
    r = r0
    flag = false
    obj_max = Inf
    witnesses = copy(witnesses_init)
    dict_indexes = Dict([(w, (i,)) for (i, w) in enumerate(witnesses)])
    get_indexes(witness) = get(dict_indexes, witness, (0,))
    coeffs_cube = (ϵ/2).*make_hypercube(Val(D))

    while true
        if iter_max ≥ 0 && iter ≥ iter_max
            @printf("Max iter (%d) exceeded\n", iter_max)
            flag = false
            break
        end
        iter += 1

        M = length(witnesses)
        _, coeffs, G, r, flag = learn_PLF_params(M, witnesses, get_indexes,
                                                 G, Gmax, r, rmin, ϵ,
                                                 solver, output=learner_output)

        if do_trace
            push!(trace.witnesses_list, copy(witnesses))
            push!(trace.coeffs_list, copy(coeffs))
            push!(trace.flags_learner, flag)
        end

        !flag && break

        append!(coeffs, coeffs_cube)
        obj_max, x, flag, i, q, σ = verify_PLF(systems, coeffs, ϵ, solver)

        if do_trace
            push!(trace.flags_verifier, flag)
        end

        !flag && break
        
        if output_period ≥ 0 && mod(iter - 1, output_period) == 0
            @printf("Iter: %d, obj_max: %f\n", iter, obj_max)
        end
        
        obj_max < tol && break

        witness = Witness(x, map(A -> A*x, systems[q].fields))
        if do_trace
            push!(trace.counterexample_list, witness)
        end
        push!(witnesses, witness)
        dict_indexes[witness] = (M + 1,)
    end

    @printf("\nTerminated (flag: %s): Iter: %d, deriv_max: %f\n",
        flag, iter, obj_max)

    return coeffs, witnesses, obj_max, flag, trace
end