module ExampleIllustrative

using LinearAlgebra
using JuMP
using Gurobi
using PyPlot

include("../../../src/CEGISPolyhedralLyapunov.jl")
CPLA = CEGISPolyhedralLyapunov.AdaptiveComplexity
CPLP = CEGISPolyhedralLyapunov.Polyhedra
include("../../utils/geometry.jl")

const GUROBI_ENV = Gurobi.Env()
solver = optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
)

## Parameters
ϵ = 50.0
θ = 1/64.0
δ = 0.001
nvar = 2

α = 6.0
sys = CPLA.System()

domain = CPLP.Cone()
A = [0.0 1.0; -2.0 -1.0]
CPLA.add_piece!(sys, domain, A)

domain = CPLP.Cone()
B = [0.0 0.0; -1.0 0.0]
CPLA.add_piece!(sys, domain, A + α*B)

## Learner feasible illustration
lear = CPLA.Learner(nvar, sys, ϵ, θ, δ)
CPLA.set_tol!(lear, :rad, 1e-10)

points_init = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
for point in points_init
    CPLA.add_point_init!(lear, point)
end

sol = CPLA.learn_lyapunov!(lear, 100, solver)

@assert sol.status == CPLA.LYAPUNOV_FOUND

end # module