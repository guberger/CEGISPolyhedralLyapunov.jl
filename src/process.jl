##------------------------------------------------------------------------------
## Adaptive

function process_PLF_adaptive(dim, systems, flows_init,
                              G0, Gmax, r0, rmin, ϵ, tol,
                              solver; kwargs...)
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
    M0 = length(coeffs_cube)
    ζ = 4/ϵ
    coeffs = coeffs_cube

    while true
        if iter_max ≥ 0 && iter ≥ iter_max
            @printf("Max iter (%d) exceeded\n", iter_max)
            flag = false
            break
        end
        iter += 1

        M = length(flows)
        M1old = length(coeffs)
        M1 = M + M0
        resize!(coeffs, M1)
        for i = M1old+1:M1
            coeffs[i] = _VT_(undef, dim)
        end

        δ, G, r, flag = learn_PLF_adaptive!(M0, M, dim, coeffs,
                                            flows, G, Gmax, r, rmin,
                                            ϵ, solver, output=learner_output)

        if do_trace
            push!(trace.flows_list, copy(flows))
            push!(trace.coeffs_list, copy(coeffs))
            push!(trace.flags_learner, flag)
        end

        !flag && break

        if output_period ≥ 0 && mod(iter, output_period) == 0
            @printf("Iter: %d, G: %f, r: %f\n", iter, G, r)
        end

        x = _VT_(undef, dim)
        obj_max, flag, i, q, σ = verify_PLF!(M1, dim, x, systems, coeffs,
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

##------------------------------------------------------------------------------
## Fixed

function process_PLF_fixed(meth,
                           M, dim, systems, seeds_init,
                           ϵ, tol, δ_min, solvers...; kwargs...)
    output_depth = get(kwargs, :output_depth, 1)
    learner_output = get(kwargs, :learner_output, true)
    depth_max = get(kwargs, :depth_max, -1)
    solverLP = solvers[1]
    solverM = get(solvers, 2, solverLP)

    depth_rec = 0 # record of max reached depth
    flag = false
    iter = 0
    obj_max = Inf
    nodes_queue = PriorityQueue{Tree,Float64}(Base.Order.Reverse)
    nodes = Root()
    for seed_init in seeds_init
        nodes = seed(seed_init)
        enqueue!(nodes_queue, nodes, Inf)
        # enqueue!(nodes_queue, nodes, iter)
        # iter += 1
    end

    coeffs_cube = ϵ.*hypercube(dim)
    M0 = length(coeffs_cube)
    M1 = M + M0
    coeffs = [zeros(dim) for i = 1:M1]
    for i = 1:M0
        copyto!(coeffs[i], coeffs_cube[i])
    end
    ζ = 2/ϵ
    meth_cheby = Chebyshev()
    
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

        δ0, flag = learn_PLF_fixed!(meth_cheby, M0, M, dim, coeffs,
                                    nodes, solverLP, output=learner_output)

        !flag && break
        
        if δ0 < 0
            if output_depth ≥ 0
                @printf("Infeasible branch: depth: %d, δ0: %f\n", depth, δ0)
            end
            continue
        end

        if meth != meth_cheby
            δ, flag = learn_PLF_fixed!(meth, M0, M, dim, coeffs,
                                       nodes, solverM, output=learner_output)
            flag = flag && δ ≥ 0
            !flag && break
        else
            δ = δ0
        end

        if output_depth ≥ 0 && mod(depth, output_depth) == 0
            @printf("Depth: %d, δ: %f (δ0: %f)\n", depth, δ, δ0)
        end

        x = _VT_(undef, dim)
        obj_max, flag, i, q, σ = verify_PLF!(M1, dim, x, systems, coeffs,
                                             ζ, solverLP)

        !flag && break

        if output_depth ≥ 0 && mod(depth, output_depth) == 0
            @printf("|----- obj_max: %f\n", obj_max)
        end

        obj_max < tol && break

        if δ ≥ δ_min
            flow = make_flow(systems, x)
            witness = Witness(flow, i)
            for j = M0+1:M1
                node = Node(witness, j)
                child = grow(nodes, node)
                enqueue!(nodes_queue, child, δ)
                # enqueue!(nodes_queue, child, iter)
                # iter += 1
            end
        else
            if output_depth ≥ 0
                @printf("Abort branch: δ too small: %f < %f, depth: %d\n",
                    δ, δ_min, depth)
            end
        end
    end

    @printf("\nTerminated (flag: %s): max depth: %d, deriv_max: %f\n",
        flag, depth_rec, obj_max)

    return coeffs, collect(Node, nodes), obj_max, flag
end