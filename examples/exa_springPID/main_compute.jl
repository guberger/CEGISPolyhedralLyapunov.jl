module ExampleSpringPID_Compute

using LinearAlgebra
using JuMP
using Gurobi
using PyPlot

include("../../src/CEGISPolyhedralVerification.jl")
CPV = CEGISPolyhedralVerification

datafile = "dataset_3"
include(string("./datasets/", datafile, ".jl"))

const GUROBI_ENV = Gurobi.Env()
function solver()
    model = direct_model(
        optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV))
    )
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "Method", 2)
    return model
end

# With τ = 1/32, ϵ = 50, δ = 1e-4; with nD = opnorm(A - I)
# finds PLF in 145 steps

## Parameters
τ = 1/32 # 0.039 # 1e-2
ϵ = 50.0 # 50.0
δ = 0.0001 # eps(1.0)
nvar = 3
nloc = 1

sys = CPV.System()

domain = CPV.Cone()
CPV.add_supp!(domain, [Ki, Kp, Kd])
A = [-c_aw 1.0 0.0; 0.0 0.0 1.0; 0.0 -a0 -a1]
CPV.add_piece_cont!(sys, domain, 1, A)

domain = CPV.Cone()
CPV.add_supp!(domain, [-Ki, -Kp, -Kd])
A = [0.0 1.0 0.0; 0.0 0.0 1.0; -Ki -(a0 + Kp) -(a1 + Kd)]
CPV.add_piece_cont!(sys, domain, 1, A)

lear = CPV.Learner(nvar, nloc, sys, τ, ϵ, δ)
CPV.set_tol!(lear, :rad, 1e-6) # -1e-5

## Solving
status, mpf, niter = CPV.learn_lyapunov!(lear, 1000, solver, solver)

display(status)

f = open(string(@__DIR__, "/results/", datafile, ".txt"), "w")
for lf in mpf.pfs[1].lfs
    println(f, lf.lin)
end
close(f)

end # module