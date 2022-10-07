module ExampleZelentsovsky_Compute

using LinearAlgebra
using JuMP
using Gurobi
using PyPlot

include("../../src/CEGISPolyhedralLyapunov.jl")
CPL = CEGISPolyhedralLyapunov

const GUROBI_ENV = Gurobi.Env()
function solver()
    model = direct_model(
        optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV))
    )
    set_optimizer_attribute(model, "OutputFlag", 0)
    return model
end

## Parameters
datafile = "dataset_1"
include(string("./datasets/", datafile, ".jl"))

A = [0.0 1.0; -2.0 -1.0]
B = [0.0 0.0; -1.0 0.0]

pieces = [
    CPL.Piece(A, Vector{Float64}[]),
    CPL.Piece(A + α*B, Vector{Float64}[])
]

lfs_init = [[0.02, 0.0], [-0.02, 0.0], [0.0, 0.02], [0.0, -0.02]]

tol_r = 1e-5
xmax = 1e4
iter_max = 500

status, lfs = CPL.learn_lyapunov(
    pieces, lfs_init, τ, 2, xmax, iter_max, solver, tol_r=tol_r
)

display(status)

f = open(string(@__DIR__, "/results/", datafile, ".txt"), "w")
for lf in lfs
    println(f, lf)
end
close(f)

end # module