function process_lyapunov_function(prob::CEGARProblem{D}, x_list,
                                   G0, Gmax, r0, rmin, params, solver) where D
    A_list = prob.A_list
    x_dx_list = map(x -> (x, map(A -> A*x, A_list)), x_list)
    c_list = Vector{Float64}[]
    meth_learn = prob.meth_learn
    meth_verify = prob.meth_verify

    # Trace
    do_trace = haskey(params, :do_trace) && params.do_trace
    c_T = Vector{Float16}
    x_dx_T = Tuple{Vector{Float64},Vector{Vector{Float64}}}
    trace = (c_list=Vector{c_T}[],
             x_dx_list=Vector{x_dx_T}[],
             flag_learner=Bool[],
             x_dx=x_dx_T[],
             flag_verifier=Bool[])

    iter = 0
    G = G0
    r = r0
    flag = false
    obj_max = Inf

    while true
        if haskey(params, :iter_max) && iter ≥ params.iter_max
            @printf("Max iter (%d) exceeded\n", params.iter_max)
            flag = false
            break
        end
        iter += 1

        _, c_list, G, r, flag = learn_candidate_lyapunov_function(
            meth_learn, x_dx_list, G, Gmax, r, rmin,
            params.tol_faces, params.print_period_1, solver)

        if do_trace
            push!(trace.x_dx_list, copy(x_dx_list))
            push!(trace.c_list, copy(c_list))
            push!(trace.flag_learner, flag)
        end

        !flag && break

        obj_max, x, flag = verify_candidate_lyapunov_function(
            meth_verify, A_list, c_list, params.tol_faces, solver)

        if do_trace
            push!(trace.flag_verifier, flag)
        end

        !flag && break
        
        if mod(iter - 1, params.print_period_2) == 0
            @printf("Iter: %d, obj_max: %f\n", iter, obj_max)
        end
        
        obj_max < params.tol_deriv && break

        x_dx = (x, map(A -> A*x, A_list))
        if do_trace
            push!(trace.x_dx, x_dx)
        end
        push!(x_dx_list, x_dx)
    end

    @printf("\nTerminated (flag: %s): Iter: %d, deriv_max: %f\n",
        flag, iter, obj_max)

    return c_list, x_dx_list, obj_max, flag, trace
end