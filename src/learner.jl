"""
    learn_candidate_lyapunov_function(method,
                                      x_dx_list, G0, Gmax, r0, rmin,
                                      tol_faces, print_period, solver)

`x_dx_list` is a vector of pairs `(x, dx_list)` where `x` is a point in the
state space and `dx_list` is a set of flow directions at `x`.
"""
function learn_candidate_lyapunov_function(method::LearnMethod,
                                           x_dx_list, G0, Gmax, r0, rmin,
                                           tol_faces, print_period, solver)
    D = state_dim(method)
    N = length(x_dx_list)
    M = -1
    if method isa LearnPolyhedralPoints
        M = N
    elseif method isa LearnPolyhedralFixed
        M = min(method.n_piece, N)
    end
    G = G0
    r = r0

    if iszero(M)
        println("Empty learning problem")
        return Inf, vec_type[], G, r, true # δ, c_list, G, r, flag
    end

    δ = -1.0
    c_list = vec_type[]
    s_status = ("Not started", "Unknown", "Unknown")
    iter = 0
    flag = false

    while G ≤ Gmax && r ≥ rmin
        iter += 1
        if print_period ≥ 0 && mod(iter - 1, print_period) == 0
            @printf("iter: %d. G: %f, r: %f\n", iter, G, r)
        end
        δ, c_list, status = _learn_polyhedral(method, M, x_dx_list,
                                              G, tol_faces, solver)
        s_status = string.(status)
        if print_period ≥ 0 && mod(iter - 1, print_period) == 0
            @printf("\tstatus: %s. δ: %f\n", s_status, δ)
        end
        flag = isone(Int(status[2])) && δ ≥ r
        (flag || 2*G > Gmax || r/2 < rmin) && break
        G = 2*G
        r = r/2
    end

    if !flag
        println("Problem in learning Lyapunov function")
        @printf("iter: %d. G: %f, r: %f\n", iter, G, r)
        println(s_status)
        println(δ)
    end

    return δ, c_list, G, r, flag
end

function _learn_polyhedral(method::LearnPolyhedralPoints{D}, M, x_dx_list,
                           G, tol_faces, solver) where D
    N = length(x_dx_list)
    model = Model(solver)
    c_list = [@variable(model, [1:D], base_name=string("c", i),
        lower_bound=-1.0, upper_bound=1.0) for i = 1:N]
    δ = @variable(model, lower_bound=0.0)

    for i = 1:N
        xt = x_dx_list[i][1]
        nxt = norm(xt)
        x = xt/nxt
        c = c_list[i]
        if tol_faces > 0
            @constraint(model, x'*c ≥ tol_faces)
        end
        for dxt in x_dx_list[i][2]
            dx = dxt/nxt
            ndx = norm(dx)
            @constraint(model, dx'*c + ndx*δ ≤ 0)
            for j = 1:N
                j == i && continue
                d = c_list[j]
                # @constraint(model, x'*(c - d) ≥ 0)
                @constraint(model, dx'*d - G*x'*(c - d) + ndx*δ ≤ 0)
            end
        end
    end

    @objective(model, Max, δ)

    optimize!(model)

    copt_list = Vector{vec_type}(undef, N)
    if has_values(model)
        δopt = value(δ)
        for i = 1:N
            copt_list[i] = value.(c_list[i])
        end
    else
        δopt = -1.0
        for i = 1:N
            copt_list[i] = zeros(D)
        end
    end

    return δopt, copt_list,
        (termination_status(model),
         primal_status(model),
         dual_status(model))
end

function _learn_polyhedral(method::LearnPolyhedralFixed{D}, M, x_dx_list,
                           G, tol_faces, solver) where D
    N = length(x_dx_list)
    model = Model(solver)
    c_list = [@variable(model, [1:D], base_name=string("c", j),
        lower_bound=-1.0, upper_bound=1.0) for j = 1:M]
    val_list = [@variable(model, base_name=string("val", i),
        lower_bound=tol_faces) for i = 1:N]
    bin_list_list = [@variable(model, [1:M], base_name=string("bin", i),
        binary=true) for i = 1:N]
    δ = @variable(model, lower_bound=0.0)

    # |x'*c| ≤ sqrt(D)
    # Hence val ≤ sqrt(D)
    # Hence val - x'*c ≤ 2*sqrt(D)
    BIG_M_x = 4*sqrt(D)
    # |dx'*c| ≤ sqrt(D)*norm(dx)
    # Hence δ ≤ sqrt(D)
    # Hence dx'*c + ndx*δ ≤ 2*sqrt(D)*norm(dx)
    # BIG_M_dx = 4*sqrt(D)

    for i = 1:N
        val = val_list[i]
        bin_list = bin_list_list[i]
        @constraint(model, sum(bin_list) == 1)
        xt, dxt = x_dx_list[i]
        nxt = norm(xt)
        x = xt/nxt
        for j = 1:M
            c = c_list[j]
            # @constraint(model, x'*c ≤ val)
            @constraint(model, val - x'*c ≤ BIG_M_x*(1 - bin_list[j]))
        end
        for dxt in x_dx_list[i][2]
            dx = dxt/nxt
            ndx = norm(dx)
            for j = 1:M
                c = c_list[j]
                @constraint(model, dx'*c + ndx*δ ≤ G*(val - x'*c))
            end
        end
    end
    
    @objective(model, Max, δ)
    
    optimize!(model)

    copt_list = Vector{vec_type}(undef, M)
    if has_values(model)
        δopt = value(δ)
        for i = 1:M
            copt_list[i] = value.(c_list[i])
        end
    else
        δopt = -1.0
        for i = 1:M
            copt_list[i] = zeros(D)
        end
    end

    return δopt, copt_list,
        (termination_status(model),
         primal_status(model),
         dual_status(model))
end