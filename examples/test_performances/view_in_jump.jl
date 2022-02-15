using JuMP
using Gurobi

function test_alloc(A, b, D, M, N)
    model = Model(solver)
    x = @variable(model, [1:D])
    r = @variable(model)
    for i = 1:M
        @constraint(model, A[(i - 1)*N+1:i*N, :]*x - b[(i - 1)*N+1:i*N] .≤ r)
    end
    @objective(model, Max, r)
    optimize!(model)
    return value(r)
end

function test_view(A, b, D, M, N)
    model = Model(solver)
    x = @variable(model, [1:D])
    r = @variable(model)
    for i = 1:M
        Av = view(A, (i - 1)*N+1:i*N, :)
        bv = view(b, (i - 1)*N+1:i*N)
        @constraint(model, Av*x - bv .≤ r)
    end
    @objective(model, Max, r)
    optimize!(model)
    return value(r)
end

function test_perfs()
    N = 100
    M = 500
    D = 100
    A = randn(N*M, D)
    b = randn(N*M)
    test_alloc(A, b, D, M, N)
    @time test_alloc(A, b, D, M, N)
    test_view(A, b, D, M, N)
    @time test_view(A, b, D, M, N)
    return nothing
end

