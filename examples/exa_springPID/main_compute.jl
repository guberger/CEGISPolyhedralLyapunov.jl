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
solver = optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
)

# Notes:
# Performs well with: ϵ = 50.0, θ = 0.04 (G = 25.0),
# New Deriv V1 (norm(deriv, Inf)) and
# New Eccentricity V1 (pos witnesses and lie witnesses)
# (requires 77 pieces)
# Performs also ok with New Deriv V2 (opnorm(A, Inf))
# (requires 155 pieces)

## Parameters
τ = 0.04 # 0.039 # 1e-2
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

for k = 1:nvar
    local point = [(k_ == k ? 1.0 : 0.0) for k_ = 1:nvar]
    CPV.add_witness!(lear, 1, point)
    CPV.add_witness!(lear, 1, -point)
end

## Solving
status = CPV.learn_lyapunov!(lear, 1000, solver)[1]

display(status)

# f = open(string(@__DIR__, "/results/", datafile, ".txt"), "w")
# for lf in sol.lfs_list[sol.niter]
#     println(f, lf.lin)
# end
# close(f)

end # module