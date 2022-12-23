module ExampleSpringPID_Compute

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
    set_optimizer_attribute(model, "Method", 2)
    return model
end

## Parameters
τ = 1/32
γ = 0.999

datafile = "dataset_3"
include(string("./datasets/", datafile, ".jl"))

pieces = [
    CPL.Piece(
        [-c_aw 1.0 0.0; 0.0 0.0 1.0; 0.0 -a0 -a1],
        [[Ki, Kp, Kd]]
    ),
    CPL.Piece(
        [0.0 1.0 0.0; 0.0 0.0 1.0; -Ki -(a0 + Kp) -(a1 + Kd)],
        [[-Ki, -Kp, -Kd]]
    )
]

lfs_init = [
    [0.02, 0.0, 0.0], [-0.02, 0.0, 0.0],
    [0.0, 0.02, 0.0], [0.0, -0.02, 0.0],
    [0.0, 0.0, 0.02], [0.0, 0.0, -0.02]
]

tol_r = 1e-6
xmax = 1e4
iter_max = 200

status, lfs = CPL.learn_lyapunov(
    pieces, lfs_init, τ, γ, 3, xmax, iter_max, solver, tol_r=tol_r
)

display(status)

f = open(string(@__DIR__, "/results/", datafile, ".txt"), "w")
for lf in lfs
    println(f, lf)
end
close(f)

end # module