function verify_single(A, lf_max, lfs_other, lfs_dom, N, xmax, solver)
    model = solver()
    x = @variable(model, [1:N], lower_bound=-xmax, upper_bound=xmax)

    for lf_other in lfs_other
        @constraint(model, dot(lf_other, x) ≤ 1)
    end

    @constraint(model, dot(lf_max, x) == 1)

    for lf_dom in lfs_dom
        @constraint(model, dot(lf_dom, x) ≤ 0)
    end

    @objective(model, Max, dot(lf_max, A*x))

    optimize!(model)

    TS = termination_status(model)
    PS = primal_status(model)

    if TS == OPTIMAL && PS == FEASIBLE_POINT
        return value.(x), objective_value(model), true
    end

    @assert TS == INFEASIBLE

    return fill(NaN, N), -Inf, false
end

function verify(
        pieces::AbstractVector{<:Piece},
        lfs::AbstractVector{<:AbstractVector},
        N, xmax, solver
    )
    xopt::Vector{Float64} = fill(NaN, N)
    γopt::Float64 = -Inf
    qopt::Int = 0
    flag_feas::Bool = false
    for (q, piece) in enumerate(pieces)
        for lf_max in lfs
            x, γ, flag = verify_single(
                piece.A, lf_max, lfs, piece.lfs_dom, N, xmax, solver
            )
            flag_feas |= flag
            if flag && γ > γopt
                xopt = x
                γopt = γ
                qopt = q
            end
        end
    end
    return xopt, γopt, qopt, flag_feas
end