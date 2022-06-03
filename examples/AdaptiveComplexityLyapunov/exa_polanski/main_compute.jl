module ExampleZelentsowsky_Compute

using LinearAlgebra
using JuMP
using Gurobi
using PyPlot

include("../../../src/CEGISPolyhedralVerification.jl")
CPLA = CEGISPolyhedralVerification.AdaptiveComplexityLyapunov
CPLP = CEGISPolyhedralVerification.Polyhedra
include("../../utils/geometry.jl")

const GUROBI_ENV = Gurobi.Env()
solver = optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
)

datafile = "dataset_2"
include(string("./datasets/", datafile, ".jl"))

## Parameters
nvar = 3

sys = CPLA.System()

domain = CPLP.Cone()
A = [-10.0 -2.0 -2.0; 1.0 0.0 0.0; 0.0 1.0 0.0]
CPLA.add_piece!(sys, domain, A)

domain = CPLP.Cone()
A = [-10.0 -10.0*α -10.0*α; 1.0 0.0 0.0; 0.0 1.0 0.0]
CPLA.add_piece!(sys, domain, A)

## Learner feasible illustration
lear = CPLA.Learner(nvar, sys, ϵ, θ, δ)
CPLA.set_tol!(lear, :rad, 1e-6)

points_init = [
    [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0], [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]
]
for point in points_init
    CPLA.add_point_init!(lear, point)
end

## Solving
sol = CPLA.learn_lyapunov!(lear, 1000, solver)

display(sol.status)

f = open(string(@__DIR__, "/results/", datafile, ".txt"), "w")
for vec in sol.vecs_list[sol.niter]
    println(f, vec)
end
close(f)

end # module