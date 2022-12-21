# α : sensitivity of lf_y'*(x + τ*A*x) - lf_x'*x
struct Witness{XT<:AbstractVector,AT<:Real,YT<:AbstractVector}
    x::XT
    α::AT
    y::YT
end

function compute_lfs(
        wit_cls::Vector{<:Vector{<:Witness}},
        lfs_init::Vector{<:AbstractVector},
        γ, N, rmax, solver
    )
    model = solver()
    lfs = [
        @variable(model, [1:N], lower_bound=-1, upper_bound=1)
        for i in eachindex(wit_cls)
    ]
    r = @variable(model, upper_bound=rmax)

    for (i, wit_cl) in enumerate(wit_cls)
        for wit in wit_cl
            valx = dot(lfs[i], wit.x)
            # do not use Iterators.flatten because type-unstable
            for lf in lfs
                valy = dot(lf, wit.y)
                @constraint(model, valy + r*wit.α ≤ γ*valx)
            end
            for lf in lfs_init
                valy = dot(lf, wit.y)
                @constraint(model, valy + r*wit.α ≤ γ*valx)
            end
        end
    end

    @objective(model, Max, r)

    optimize!(model)

    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT

    return map(lf -> map(value, lf), lfs), objective_value(model)
end