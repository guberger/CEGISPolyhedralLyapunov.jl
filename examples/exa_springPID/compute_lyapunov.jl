module ExampleSpringPID_ComputeLyapunov

using JuMP
using Gurobi
using PyPlot
include("../../src/CEGPolyhedralLyapunov.jl")
CPL = CEGPolyhedralLyapunov

datafile = "data_set_3"
include(string("./", datafile, ".jl"))

const GUROBI_ENV = Gurobi.Env()
solver = optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV),
    "OutputFlag"=>false)

## Parameters
domain1 = [+Ki +Kp +Kd]
fields1 = [[-c_aw 1.0 0.0; 0.0 0.0 1.0; 0.0 -a0 -a1]]
domain2 = [-Ki -Kp -Kd]
fields2 = [[0.0 1.0 0.0; 0.0 0.0 1.0; -Ki -(a0 + Kp) -(a1 + Kd)]]
sys1 = CPL.LinearSystem(domain1, fields1)
sys2 = CPL.LinearSystem(domain2, fields2)
systems = (sys1, sys2)
D = 3

G0 = 0.1
Gmax = 1000.0
r0 = 0.01
rmin = eps(1.0)
ϵ = 1/50
tol = -1e-5

points = CPL.make_hypercube(D)
witnesses_init = CPL.make_witnesses(systems, points)

## Solving
coeffs, witnesses, deriv, flag, trace =
    @time CPL.process_PLF(D, systems, witnesses_init,
                          G0, Gmax, r0, rmin, ϵ, tol,
                          solver)

f = open(string(@__DIR__, "/lyapunov-", datafile, ".txt"), "w")
for c in coeffs
    println(f, c)
end
close(f)

end # module