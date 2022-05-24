module ExampleSpringPID_Compute

using JuMP
using Gurobi
using PyPlot

include("../../../src/CEGISPolyhedralLyapunov.jl")
CPL = CEGISPolyhedralLyapunov
CPLA = CPL.AdaptiveComplexity
CPLP = CPL.Polyhedra

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
ϵ = 50.0 # 50.0
θ = 0.04 # 0.039 # 1e-2
δ = 0.0001 # eps(1.0)
nvar = 3
prob = CPLA.LearningProblem(nvar, ϵ, θ, δ)
CPLA.set_tol_rad!(prob, 1e-6) # -1e-5

# α = 1.5
# CPLA.set_Gs!(prob, α)
CPLA.add_G!(prob, 1/θ)

domain = CPLP.Cone()
CPLP.add_supp!(domain, CPLP.Supp([Ki, Kp, Kd]))
A = [-c_aw 1.0 0.0; 0.0 0.0 1.0; 0.0 -a0 -a1]
CPLA.add_system!(prob, domain, A)

domain = CPLP.Cone()
CPLP.add_supp!(domain, CPLP.Supp([-Ki, -Kp, -Kd]))
A = [0.0 1.0 0.0; 0.0 0.0 1.0; -Ki -(a0 + Kp) -(a1 + Kd)]
CPLA.add_system!(prob, domain, A)

for k = 1:nvar
    local point = [(k_ == k ? 1.0 : 0.0) for k_ = 1:nvar]
    CPLA.add_point_init!(prob, point)
    CPLA.add_point_init!(prob, -point)
end

## Solving
sol = CPLA.learn_lyapunov!(prob, 1000, solver)

display(sol.status)

# coeffs, flows, deriv, flag, trace =
#     CPL.process_PLF_adaptive(D, systems, flows_init,
#                              G0, Gmax, r0, rmin, ϵ, tol,
#                              solver, output_period=10, learner_output=false)

# f = open(string(@__DIR__, "/lyapunov-", datafile, ".txt"), "w")
# for c in coeffs
#     println(f, c)
# end
# close(f)

end # module