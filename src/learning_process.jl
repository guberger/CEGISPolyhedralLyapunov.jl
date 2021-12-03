function learning_clf_process(method, A_list, params, solver)
    D = state_dim(method)
    N = Int(get_def(params, :N, 2*D))
    tol_faces = Float64(get_def(params, :tol_faces, D*1e-6))
    tol_derivative = Float64(get_def(params, :tol_derivative, D*1e-3))
    print_period = Int(get_def(params, :print_period, 1000))

    x_list = [randn(D) for i = 1:N]
    x_dx_list = map(x -> (x, map(A -> A*x, A_list)), x_list)

    out_msg = "ok"
    info_msg = ""
    c_list = Vector{Vector{Float64}}[]
    iter = 0
    obj_max = -Inf

    while true
        if haskey(params, :iter_max) && iter ≥ params.iter_max
            @printf("Max iter (%d) exceeded\n", params.iter_max)
            break
        end
        iter += 1
        obj_max = -Inf

        r, c_list, status = learn_candidate_lyapunov_function(
            method, x_dx_list, params.Gain, solver)

        if Int(status.primal) != 1 || r < 0
            println("Problem in learning Lyapunov function")
            println(status.termination)
            println(status.primal)
            println(status.dual)
            @printf("obj: %f\n", obj)
            break
        end

        if any(c -> norm(c) < tol_faces, c_list)
            println("Polyhedral function near to zero")
            @printf("Minimum norm: %f\n", minimum(c -> norm(c), c_list))
            break
        end

        output_list = verify_candidate_lyapunov_function(
            method, A_list, c_list, solver)
        n_new = 0

        for output in output_list
            obj, x, status = output

            if Int(status.primal) != 1 || isinf(obj)
                println("Problem in verifying Lyapunov function")
                println(status.termination)
                println(status.primal)
                println(status.dual)
                @printf("obj: %f\n", obj)
                break
            end

            obj_max = max(obj_max, obj)

            obj < tol_derivative && continue

            x_dx = (x, map(A -> A*x, A_list))
            push!(x_dx_list, x_dx)
            n_new += 1
        end

        if mod(iter - 1, print_period) == 0
            @printf("Iter: %d, obj_max: %f\n", iter, obj_max)
        end

        iszero(n_new) && break
    end

    @printf("\nTerminated: Iter: %d, obj_max: %f\n", iter, obj_max)

    return c_list, x_dx_list
end