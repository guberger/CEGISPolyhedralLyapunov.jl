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
    coeffs_cube = (ϵ/2).*hypercube(dim)
    ζ = 4/ϵ

    while true
        if iter_max ≥ 0 && iter ≥ iter_max
            @printf("Max iter (%d) exceeded\n", iter_max)
            flag = false
            break
        end
        iter += 1

        δ, coeffs, G, r, flag = learn_PLF_adaptive(dim, flows,
                                                   G, Gmax, r, rmin, ϵ,
                                                   solver,
                                                   output=learner_output)

        if do_trace
            push!(trace.flows_list, copy(flows))
            push!(trace.coeffs_list, copy(coeffs))
            push!(trace.flags_learner, flag)
        end

        !flag && break

        if output_period ≥ 0 && mod(iter, output_period) == 0
            @printf("Iter: %d, G: %f, r: %f\n", iter, G, r)
        end

        append!(coeffs, coeffs_cube)
        M = length(coeffs)
        obj_max, x, flag, i, q, σ = verify_PLF(M, dim, systems, coeffs,
                                               ζ, solver)

        if do_trace
            push!(trace.flags_verifier, flag)
        end

        !flag && break
        
        if output_period ≥ 0 && mod(iter, output_period) == 0
            @printf("|---- obj_max: %f\n", obj_max)
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

function process_PLF_fixed(M, dim, systems, nodes_init,
                           ϵ, tol, δ_min, solver; kwargs...)
    coeffs = [zeros(dim) for i = 1:M]
    output_depth = get(kwargs, :output_depth, 1)
    learner_output = get(kwargs, :learner_output, true)
    depth_max = get(kwargs, :depth_max, -1)

    depth_rec = 0 # record of max reached depth
    flag = false
    obj_max = Inf
    nodes_queue = PriorityQueue{Tree,Float64}(Base.Order.Reverse)
    nodes = seed(nodes_init)
    enqueue!(nodes_queue, nodes, Inf)
    coeffs_cube = ϵ.*hypercube(dim)
    append!(coeffs, coeffs_cube)
    N = length(coeffs)
    ζ = 2/ϵ
    
    while !isempty(nodes_queue)
        nodes = dequeue!(nodes_queue)
        depth = length(nodes)
        if depth_max ≥ 0 && depth > depth_max
            if output_depth ≥ 0
                @printf("Abort branch: max depth (%d) exceeded\n", depth_max)
            end
            continue
        end
        depth_rec = max(depth_rec, depth)

        δ, flag = learn_PLF_fixed!(M, dim, coeffs, nodes,
                                   solver, output=learner_output)

        !flag && break

        if output_depth ≥ 0 && mod(depth, output_depth) == 0
            @printf("Depth: %d, δ: %f\n", depth, δ)
        end
        
        if δ < 0
            if output_depth ≥ 0
                @printf("Infeasible branch: depth: %d, δ: %f\n", depth, δ)
            end
            continue
        end

        obj_max, x, flag, i, q, σ = verify_PLF(N, dim, systems, coeffs,
                                               ζ, solver)

        !flag && break

        if output_depth ≥ 0 && mod(depth, output_depth) == 0
            @printf("|---- obj_max: %f\n", obj_max)
        end

        obj_max < tol && break

        if δ ≥ δ_min
            flow = make_flow(systems, x)
            witness = Witness(flow, i)
            for i = 1:M
                node = Node(witness, i)
                child = grow(nodes, node)
                enqueue!(nodes_queue, child, δ)
            end
        else
            if output_depth ≥ 0
                @printf("Abort branch: δ too small: %f < %f\n", δ, δ_min)
            end
        end
    end

    @printf("\nTerminated (flag: %s): max depth: %d, deriv_max: %f\n",
        flag, depth_rec, obj_max)

    return coeffs, nodes, obj_max, flag
end