using LinearAlgebra
using JuMP
using MosekTools

A1 = [-0.2 1.0; -1.0 -0.2]
A2 = [-0.2 0.0; -0.5 -0.2]

solver = MosekTools.Optimizer
model = Model(solver)

P = @variable(model, [1:2, 1:2], Symmetric)
@constraint(model, P - I in PSDCone())
@constraint(model, -I - (A1'*P + P*A1) in PSDCone())
@constraint(model, -I - (A2'*P + P*A2) in PSDCone())
@objective(model, Min, tr(P))

optimize!(model)

display(solution_summary(model))
display(value.(P))
