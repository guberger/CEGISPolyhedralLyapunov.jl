function process_lyapunov_function(prob::CEGARProblem{D,LT,VT},
                                   x_list, G0, Gmax, r0, rmin, params, solver
                                   ) where {D,LT<:Polyhedral,VT<:Polyhedral}
    A_list = prob.A_list
    x_dx_list = map(x -> (x, map(A -> A*x, A_list)), x_list)
    c_list = Vector{Float64}[]
    lt_ = LT()
    vt_ = VT()

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
            lt_, x_dx_list, G, Gmax, r, rmin,
            params.tol_faces, params.print_period_1, solver)

        !flag && break

        obj_max, x, flag = verify_candidate_lyapunov_function(
            vt_, A_list, c_list, params.tol_faces, solver)

        !flag && break
        
        if mod(iter - 1, params.print_period_2) == 0
            @printf("Iter: %d, obj_max: %f\n", iter, obj_max)
        end
        
        obj_max < params.tol_deriv && break

        x_dx = (x, map(A -> A*x, A_list))
        push!(x_dx_list, x_dx)
    end

    @printf("\nTerminated (flag: %s): Iter: %d, deriv_max: %f\n",
        flag, iter, obj_max)

    return c_list, x_dx_list, obj_max, flag
end