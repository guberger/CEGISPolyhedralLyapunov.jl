function process_PLF_adaptive(dim, systems, flows_init,
                              G0, Gmax, r0, rmin, ϵ, tol,
                              solver; kwargs...)
    coeffs = _VT_[]
    output_period = get(kwargs, :output_period, 1)
    learner_output = get(kwargs, :learner_output, true)
    iter_max = get(kwargs, :iter_max, -1)
    do_trace = get(kwargs, :trace, true)

    # Trace
    trace = (coeffs_list=Vector{_VT_}[],
             flows_list=Vector{Flow}[],
             flags_learner=Bool[],
             counterexample_list=Flow[],
             flags_verifier=Bool[])

    iter = 0
    G = G0
    r = r0
    flag = false
    obj_max = Inf
    flows = collect(Flow, flows_init)
    coeffs_cube = (ϵ/2).*make_hypercube(dim)

    while true
        if iter_max ≥ 0 && iter ≥ iter_max
            @printf("Max iter (%d) exceeded\n", iter_max)
            flag = false
            break
        end
        iter += 1

        _, coeffs, G, r, flag = learn_PLF_robust(dim, flows,
                                                 G, Gmax, r, rmin, ϵ,
                                                 solver, output=learner_output)

        if do_trace
            push!(trace.flows_list, copy(flows))
            push!(trace.coeffs_list, copy(coeffs))
            push!(trace.flags_learner, flag)
        end

        !flag && break

        append!(coeffs, coeffs_cube)
        obj_max, x, flag, i, q, σ = verify_PLF(dim, systems, coeffs, ϵ, solver)

        if do_trace
            push!(trace.flags_verifier, flag)
        end

        !flag && break
        
        if output_period ≥ 0 && mod(iter - 1, output_period) == 0
            @printf("Iter: %d, obj_max: %f\n", iter, obj_max)
        end
        
        obj_max < tol && break

        flow = Flow(x, map(A -> A*x, systems[q].fields))
        if do_trace
            push!(trace.counterexample_list, flow)
        end
        push!(flows, flow)
    end

    @printf("\nTerminated (flag: %s): Iter: %d, deriv_max: %f\n",
        flag, iter, obj_max)

    return coeffs, flows, obj_max, flag, trace
end

function process_PLF_fixed(M, dim, nodes_init, ϵ, tol, solver; kwargs...)
    coeffs = [zeros(D) for i = 1:M]
    output_period = get(kwargs, :output_period, 1)
    learner_output = get(kwargs, :learner_output, true)
    iter_max = get(kwargs, :iter_max, -1)
    depth_max = get(kwargs, :depth_max, -1)

    nodes_stack = [linked_collection(Node, nodes_init)]


end