module TestMain

using LinearAlgebra
using Printf
using JuMP
using Gurobi
include("../../src/CEGARLearningCLF.jl")
CLC = CEGARLearningCLF

## Parameters
G0 = 0.1
Gmax = 10.0
r0 = 0.01
rmin = 1e-6
params = (tol_faces=1e-2, tol_deriv=-1e-5,
          print_period_1=-1, print_period_2=10,
          do_trace=false)
const GUROBI_ENV = Gurobi.Env()
solver = optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV),
    "OutputFlag"=>false)

HP(D) = [1.0, zeros(D - 1)...]

f = open(string(@__DIR__, "/measurements.txt"), "w")
iter = 0
max_iter = 50

for D in (4, 5, 6, 7, 8, 9)
    iter > max_iter && break
    meth_learn = CLC.LearnPolyhedralPoints{D}()
    meth_verify = CLC.VerifyPolyhedralMultiple{D}()
    # f_M = x -> sin(sqrt(x)) - log(x)
    # M = reshape(map(f_M, 1:D*D), D, D)
    M = Matrix{Float64}(I, D, D)
    M[:, 1] = 1:D
    U = qr(M).Q
    Hs1 = [+HP(D)]
    Hs2 = [-HP(D)]
    Hs_list = [Hs1, Hs2]
    str = @sprintf("D = %d\n", D)
    print(string("---> ", str))
    print(f, str)
    for γ in (1, 0.1, 0.01)
        global iter += 1
        iter > max_iter && break
        A1 = U \ (ones(D, D) - (D + γ)*Matrix{Bool}(I, D, D)) * U
        A2 = U \ (ones(D, D) - (D + 1)*Matrix{Bool}(I, D, D)) * U
        As_list = [[A1], [A2]]
        sys = CLC.PiecewiseLinearSystem{D}(2, Hs_list, As_list)
        prob = CLC.CEGARProblem(sys, meth_learn, meth_verify)
        x_list = CLC.vec_type[]
        @time output = CLC.process_lyapunov_function(
            prob, x_list, G0, Gmax, r0, rmin, params, solver)
        time = @elapsed CLC.process_lyapunov_function(
            prob, x_list, G0, Gmax, r0, rmin, params, solver)
        complexity = length(output[1])
        flag = output[4]
        deriv = output[3]
        σ = -maximum(real.(eigvals(A1)))
        str = @sprintf("%s %f | %f & %e & %.2f & %d\n",
            flag, deriv, γ, σ, time, complexity)
        print(string("---> ", str))
        print(f, str)
    end
end

close(f)

end # TestMain