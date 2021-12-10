function learn_candidate_lyapunov_function(method::LearnPolyhedralPoints,
                                           x_dx_list, G0, Gmax, r0, rmin,
                                           tol_faces, print_period, solver)
    D = state_dim(method)
    N = length(x_dx_list)
    G = G0
    r = r0

    δ = -1.0
    c_list = Vector{Vector{Float64}}(undef, N)
    for i = 1:N
        c_list[i] = zeros(D)
    end
    s_status = ("Not started", "Unknown", "Unknown")
    iter = 0
    flag = false

    while G ≤ Gmax && r ≥ rmin
        iter += 1
        if mod(iter - 1, print_period) == 0
            @printf("iter: %d. G: %f\n", iter, G)
        end
        δ, c_list, status = _learn_polyhedralpoints(D, N, x_dx_list,
                                                    G, tol_faces, solver)
        s_status = string.(status)
        if mod(iter - 1, print_period) == 0
            @printf("\tstatus: %s. δ: %f\n", s_status, δ)
        end
        flag = isone(Int(status[2])) && δ ≥ r
        (flag || 2*G > Gmax || r/2 < rmin) && break
        G = 2*G
        r = r/2
    end

    if !flag
        println("Problem in learning Lyapunov function")
        @printf("iter: %d. G: %f\n", iter, G)
        println(s_status)
        println(δ)
    end

    return δ, c_list, G, r, flag
end

function _learn_polyhedralpoints(D, N, x_dx_list, G, tol_faces, solver)
    model = Model(solver)
    c_list = [@variable(model, [1:D], base_name=string("c", i),
        lower_bound=-1.0, upper_bound=1.0) for i = 1:N]
    δ = @variable(model, lower_bound=0.0)

    for i = 1:N
        xt = x_dx_list[i][1]
        nxt = norm(xt)
        x = xt/nxt
        c = c_list[i]
        for dxt in x_dx_list[i][2]
            dx = dxt/nxt
            ndx = norm(dx)
            if tol_faces > 0
                @constraint(model, x'*c ≥ tol_faces)
            end
            @constraint(model, dx'*c + ndx*δ ≤ 0)
            for j = 1:N
                j == i && continue
                d = c_list[j]
                @constraint(model, x'*(c + d) ≥ 0)
                @constraint(model, x'*(c - d) ≥ 0)
                @constraint(model, (+dx)'*d - G*x'*(c - d) + ndx*δ ≤ 0)
                @constraint(model, (-dx)'*d - G*x'*(c + d) + ndx*δ ≤ 0)
            end
        end
    end

    @objective(model, Max, δ)

    optimize!(model)

    if has_values(model)
        δopt = value(δ)
        copt_list = map(c -> value.(c), c_list)
    else
        δopt = -1.0
        copt_list = Vector{Vector{Float64}}(undef, N)
        for i = 1:N
            copt_list[i] = zeros(D)
        end
    end

    return δopt, copt_list,
        (termination_status(model),
         primal_status(model),
         dual_status(model))
end