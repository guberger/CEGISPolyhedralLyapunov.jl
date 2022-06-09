module ExampleZelentsowsky_Compute

using LinearAlgebra
using JuMP
using Gurobi
using PyPlot

include("../../src/CEGISPolyhedralVerification.jl")
CPV = CEGISPolyhedralVerification

const GUROBI_ENV = Gurobi.Env()
function solver()
    model = direct_model(
        optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV))
    )
    set_optimizer_attribute(model, "OutputFlag", 0)
    # set_optimizer_attribute(model, "Method", 2)
    return model
end

datafile = "dataset_1"
include(string("./datasets/", datafile, ".jl"))

## Parameters
nvar = 2
nloc = 1

sys = CPV.System()

domain = CPV.Cone()
A = [0.0 1.0; -2.0 -1.0]
CPV.add_piece_cont!(sys, domain, 1, A)

domain = CPV.Cone()
B = [0.0 0.0; -1.0 0.0]
CPV.add_piece_cont!(sys, domain, 1, A + α*B)

## Learner feasible illustration
lear = CPV.Learner(nvar, nloc, sys, τ, ϵ, δ)
CPV.set_tol!(lear, :rad, 1e-6)

points_init = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
for point in points_init
    CPV.add_witness!(lear, 1, point)
end

## Solving
status, mpf, niter = CPV.learn_lyapunov!(lear, 1000, solver, solver)

display(status)

f = open(string(@__DIR__, "/results/", datafile, ".txt"), "w")
for lf in mpf.pfs[1].lfs
    println(f, lf.lin)
end
close(f)

end # module