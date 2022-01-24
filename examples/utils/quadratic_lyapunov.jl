using LinearAlgebra
using JuMP
using MosekTools

function quadratic_lyapunov(A_list)
    @assert !isempty(A_list)
    D = size(A_list[1], 1)
    @assert all(A -> size(A, 1) == D, A_list)
    @assert all(A -> size(A, 2) == D, A_list)
    
    solver = MosekTools.Optimizer
    model = Model(solver)

    P = @variable(model, [1:D, 1:D], Symmetric)
    @constraint(model, P - I in PSDCone())

    for A in A_list
        @constraint(model, -I - (A'*P + P*A) in PSDCone())
    end

    @objective(model, Min, tr(P))

    optimize!(model)

    return solution_summary(model), value.(P)
end
