using JuMP
using Gurobi

solver = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag"=>true)
N = 10

function LP_single(N, solver)
    model = Model(solver)
    x_list = [@variable(model, [1:3], lower_bound=0.0) for i = 1:N]
    r_list = [@variable(model) for i = 1:N]
    for i = 1:N
        @constraint(model, x_list[i][3] ≥ 1)
        @constraint(model, r_list[i] ≥ sum(x_list[i]))
    end
    @objective(model, Min, sum(r_list))
    optimize!(model)
    r_opt = maximum(value.(r_list))
    return r_opt
end

function LP_multiple(N, solver)
    r_opt = -Inf
    for i = 1:N
        model = Model(solver)
        x = @variable(model, [1:3], lower_bound=0.0)
        r = @variable(model)
        @constraint(model, x[3] ≥ 1)
        @constraint(model, r ≥ sum(x))
        @objective(model, Min, sum(r))
        optimize!(model)
        r_opt = max(r_opt, value(r))
    end
    return r_opt
end

LP_single(N, solver)
LP_multiple(N, solver)

