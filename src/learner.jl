function learn_candidate_lyapunov_function(method::LearnPolyhedralPoints,
                                           x_dx_list, G0, Gmax,
                                           tol_faces, print_period, solver)
    D = state_dim(method)
    N = length(x_dx_list)
    G = G0

    r = -1.0
    c_list = Vector{Vector{Float64}}(undef, N)
    for i = 1:N
        c_list[i] = zeros(D)
    end
    status = ("Not solved", "Unknown", "Unknown")
    norm_faces = 0.0
    iter = 0
    flag = false

    while G ≤ Gmax
        iter += 1
        if mod(iter - 1, print_period) == 0
            @printf("iter: %d. G: %f\n", iter, G)
        end
        r, c_list, status = _learn_polyhedralpoints(D, N, x_dx_list, G, solver)
        norm_faces = minimum(c -> norm(c), c_list)
        if mod(iter - 1, print_period) == 0
            @printf("\tstatus: %d. Norm faces: %f\n", flag, norm_faces)
        end
        flag = status[2] == "FEASIBLE_POINT" && norm_faces ≥ tol_faces
        (flag || 2*G > Gmax) && break
        G = 2*G
    end

    if !flag
        println("Problem in learning Lyapunov function")
        @printf("iter: %d. G: %f\n", iter, G)
        println(status)
        println(norm_faces)
    end

    return r, c_list, G, flag
end

function _learn_polyhedralpoints(D, N, x_dx_list, G, solver)
    model = Model(solver)
    c_list = [@variable(model, [1:D], base_name=string("c", i),
        lower_bound=-1.0, upper_bound=1.0) for i = 1:N]
    r = @variable(model)

    for i = 1:N
        xt = x_dx_list[i][1]
        nxt = norm(xt)
        x = xt/nxt
        c = c_list[i]
        for dxt in x_dx_list[i][2]
            dx = dxt/nxt
            ndx = norm(dx)
            @constraint(model, dx'*c + ndx*r ≤ 0)
            for j = 1:N
                j == i && continue
                d = c_list[j]
                @constraint(model, x'*(c + d) ≥ 0)
                @constraint(model, x'*(c - d) ≥ 0)
                @constraint(model, (+dx)'*d - G*x'*(c - d) + ndx*r ≤ 0)
                @constraint(model, (-dx)'*d - G*x'*(c + d) + ndx*r ≤ 0)
            end
        end
    end

    @objective(model, Max, r)

    optimize!(model)

    if has_values(model)
        ropt = value(r)
        copt_list = map(c -> value.(c), c_list)
    else
        ropt = -1.0
        copt_list = Vector{Vector{Float64}}(undef, N)
        for i = 1:N
            copt_list[i] = zeros(D)
        end
    end

    return ropt, copt_list,
        (string(termination_status(model)),
         string(primal_status(model)),
         string(dual_status(model)))
end