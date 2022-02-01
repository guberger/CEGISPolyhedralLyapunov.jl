function process_lyapunov_function(prob::CEGARProblem{D}, x_list,
                                   G0, Gmax, r0, rmin, params, solver) where D
    sys = prob.sys
    c_list = vec_type[]
    meth_learn = prob.meth_learn
    meth_verify = prob.meth_verify
    tol_faces = params.tol_faces
    print_period = params.print_period_2

    # Trace
    do_trace = haskey(params, :do_trace) && params.do_trace
    trace = (c_list=Vector{vec_type}[],
             x_dx_list=Vector{x_dx_type}[],
             flag_learner=Bool[],
             x_dx=x_dx_type[],
             flag_verifier=Bool[])

    iter = 0
    G = G0
    r = r0
    flag = false
    obj_max = Inf
    x_dx_list = _initial_witnesses(sys, x_list)
    c0_list = _hypercube(D, tol_faces/2)

    while true
        if haskey(params, :iter_max) && iter ≥ params.iter_max
            @printf("Max iter (%d) exceeded\n", params.iter_max)
            flag = false
            break
        end
        iter += 1

        _, c_list, G, r, flag = learn_candidate_lyapunov_function(
            meth_learn, x_dx_list, G, Gmax, r, rmin,
            tol_faces, params.print_period_1, solver)

        if do_trace
            push!(trace.x_dx_list, copy(x_dx_list))
            push!(trace.c_list, copy(c_list))
            push!(trace.flag_learner, flag)
        end

        !flag && break

        append!(c_list, c0_list)

        obj_max, x, flag, j, q = verify_candidate_lyapunov_function(
            meth_verify, sys, c_list, params.tol_faces, solver)

        if do_trace
            push!(trace.flag_verifier, flag)
        end

        !flag && break
        
        if print_period ≥ 0 && mod(iter - 1, print_period) == 0
            @printf("Iter: %d, obj_max: %f\n", iter, obj_max)
        end
        
        obj_max < params.tol_deriv && break

        x_dx = (x, map(A -> A*x, sys.As_list[q]))
        if do_trace
            push!(trace.x_dx, x_dx)
        end
        push!(x_dx_list, x_dx)
    end

    @printf("\nTerminated (flag: %s): Iter: %d, deriv_max: %f\n",
        flag, iter, obj_max)

    return c_list, x_dx_list, obj_max, flag, trace
end

function _initial_witnesses(sys, x_list)
    Q = sys.n_mode
    x_dx_list = x_dx_type[]
    for x in x_list
        for q in 1:Q
            A_set, H_set = sys.As_list[q], sys.Hs_list[q]
            any(h -> h'*x > 0, H_set) && continue
            push!(x_dx_list, (x, map(A -> A*x, A_set)))
        end
    end
    return x_dx_list
end

function _hypercube(D, ϵ)
    c_list = Vector{vec_type}(undef, 2*D)
    for i = 1:D
        c_list[2*i - 1] = vcat(zeros(i - 1), [+ϵ], zeros(D - i))
        c_list[2*i + 0] = vcat(zeros(i - 1), [-ϵ], zeros(D - i))
    end
    return c_list
end