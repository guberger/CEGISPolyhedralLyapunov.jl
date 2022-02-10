using StaticArrays
using JuMP
using Gurobi

function test_1(N, ::Val{D}, A, b, solver) where D
    model = Model(solver)
    vars = [SVector{D}(ntuple(i -> @variable(model, lower_bound=0.0), Val(D)))
            for i = 1:N]
    r = @variable(model)
    for i = 1:N
        Av = A*vars[i]
        for k = 1:D
            @constraint(model, Av[k] + r ≤ b[k])
        end
    end
    @objective(model, Max, r)
    optimize!(model)
    return value(r)
end

function test_2(N, ::Val{D}, A, b, solver) where D
    model = Model(solver)
    vars = [SVector{D}(@variable(model, [1:D], lower_bound=0.0))
            for i = 1:N]
    r = @variable(model)
    for i = 1:N
        Av = A*vars[i]
        for k = 1:D
            @constraint(model, Av[k] + r ≤ b[k])
        end
    end
    @objective(model, Max, r)
    optimize!(model)
    return value(r)
end

function test_3(N, ::Val{D}, A, b, solver) where D
    model = Model(solver)
    vars = [@variable(model, [1:D], lower_bound=0.0)
            for i = 1:N]
    r = @variable(model)
    for i = 1:N
        @constraint(model, A*vars[i] .+ r .≤ b)
    end
    @objective(model, Max, r)
    optimize!(model)
    return value(r)
end

function test_perfs()
    N = 1000
    solver = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag"=>false)
    A1 = A2 = ones(SMatrix{15,15})
    b1 = b2 = ones(SVector{15})
    A3 = ones(15, 15)
    b3 = ones(15)
    test_1(N, Val(15), A1, b1, solver)
    @time test_1(N, Val(15), A1, b1, solver)
    test_2(N, Val(15), A2, b2, solver)
    @time test_2(N, Val(15), A2, b2, solver)
    test_3(N, Val(15), A3, b3, solver)
    @time test_3(N, Val(15), A3, b3, solver)
    return nothing
end

