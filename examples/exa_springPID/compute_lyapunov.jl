module TestMain

using JuMP
using Gurobi
using PyPlot
include("../../src/CEGARLearningCLF.jl")
CLC = CEGARLearningCLF

datafile = "data_set_3"
include(string("./", datafile, ".jl"))

const GUROBI_ENV = Gurobi.Env()
solver = optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV),
    "OutputFlag"=>false)

## Parameters
meth_learn = CLC.LearnPolyhedralPoints{3}()
meth_verify = CLC.VerifyPolyhedralMultiple{3}()
Hs1 = [[+Ki, +Kp, +Kd]]
As1 = [[-c_aw 1.0 0.0; 0.0 0.0 1.0; 0.0 -a0 -a1]]
Hs2 = [[-Ki, -Kp, -Kd]]
As2 = [[0.0 1.0 0.0; 0.0 0.0 1.0; -Ki -(a0 + Kp) -(a1 + Kd)]]
Hs_list = [Hs1, Hs2]
As_list = [As1, As2]
sys = CLC.PiecewiseLinearSystem{3}(2, Hs_list, As_list)
prob = CLC.CEGARProblem(sys, meth_learn, meth_verify)
G0 = 0.1
Gmax = 1000.0
r0 = 0.01
rmin = eps(1.0)
params = (tol_faces=1/50, tol_deriv=-1e-5,
          print_period_1=1, print_period_2=1,
          do_trace=true)

x_list = CLC._hypercube(3, 1.0)

## Solving
c_list, x_dx_list, deriv, flag, trace = @time CLC.process_lyapunov_function(
    prob, x_list, G0, Gmax, r0, rmin, params, solver)

f = open(string(@__DIR__, "/lyapunov-", datafile, ".txt"), "w")
for c in c_list
    println(f, c)
end
close(f)

end # TestMain