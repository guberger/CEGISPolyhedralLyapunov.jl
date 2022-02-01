module TestMain

using LinearAlgebra
using Printf
using JuMP
using Gurobi
include("../../src/CEGARLearningCLF.jl")
CLC = CEGARLearningCLF

## Parameters
D = 5
meth_learn = CLC.LearnPolyhedralPoints{D}()
meth_verify = CLC.VerifyPolyhedralMultiple{D}()
Hs = CLC.vec_type[]
U = [-0.504806   -0.0623511   0.285222    -0.713531  -0.388338;
      0.76889    -0.357012    0.3045      -0.174628  -0.397663;
      0.107636   -0.127327   -0.908567    -0.295728  -0.243419;
      0.377144    0.659279    0.0204184   -0.524993   0.383509;
     -0.0125158  -0.64637     0.00410404  -0.31194    0.696223]
Hs_list = [Hs]
G0 = 0.1
Gmax = 10.0
r0 = 0.01
rmin = 1e-6
params = (tol_faces=1e-2, tol_deriv=-1e-5,
          print_period_1=100, print_period_2=100,
          do_trace=false)
solver = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag"=>false)

f = open(string(@__DIR__, "/measurements.txt"), "w")
iter = 0

for γ = (-4.5, -4.25, -4.125, -4.0625, -4.03125, -4.015625)
    global iter += 1
    iter > 20 && break
    A = U \ (ones(D, D) + (γ - 1)*Matrix{Bool}(I, D, D)) * U
    As_list = [[A]]
    sys = CLC.PiecewiseLinearSystem{D}(1, Hs_list, As_list)
    prob = CLC.CEGARProblem(sys, meth_learn, meth_verify)
    x_list = CLC.vec_type[]
    output = CLC.process_lyapunov_function(
        prob, x_list, G0, Gmax, r0, rmin, params, solver)
    time = @elapsed CLC.process_lyapunov_function(
        prob, x_list, G0, Gmax, r0, rmin, params, solver)
    complexity = length(output[1])
    flag = output[4]
    deriv = output[3]
    σ = (4 + γ) / (γ - 1)
    print(f, @sprintf("%s %f | %f & %e & %.2f & %d\n",
        flag, deriv, γ, σ, time, complexity))
end

close(f)

end # TestMain