module TestMain

using JuMP
using HiGHS
using Test
@static if isdefined(Main, :TestLocal)
    include("../src/CEGARLearningCLF.jl")
else
    using CEGARLearningCLF
end
CLC = CEGARLearningCLF

# Temporary fix
function HiGHS._check_ret(ret::Cint) 
    if ret != Cint(0) && ret != Cint(1)
        error(
            "Encountered an error in HiGHS (Status $(ret)). Check the log " * 
            "for details.", 
        ) 
    end 
    return 
end 

sleep(0.1) # used for good printing
println("Started test")

## Parameters
meth_learn = CLC.LearnPolyhedralPoints{2}()
meth_verify = CLC.VerifyPolyhedralMultiple{2}()
A1 = [-0.5 1.0; -1.0 -0.5]
A2 = [-0.2 0.0; -0.5 -0.2]
A_list = [A1, A2]
prob = CLC.CEGARProblem{2}(A_list, meth_learn, meth_verify)
G0 = 0.1
r0 = 0.01
rmin = 1e-6
params = (tol_faces=1e-5, tol_deriv=1e-5,
          print_period_1=1, print_period_2=1, iter_max=100)
solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)

N = 10
α_list = range(0.0, stop=2π, length=N)
x_list = map(α -> [cos(α), sin(α)], α_list)

## Solving
@testset "Process LTI feasible" begin
    Gmax = 10.0
    c_list, x_dx_list, deriv, flag = CLC.process_lyapunov_function(
        prob, x_list, G0, Gmax, r0, rmin, params, solver)
    @test deriv < 1e-9
    @test flag
end

## Solving
@testset "Process LTI infeasible G" begin
    Gmax = 1.0
    c_list, x_dx_list, deriv, flag = CLC.process_lyapunov_function(
        prob, x_list, G0, Gmax, r0, rmin, params, solver)
    @test isinf(deriv)
    @test !flag
end

# Solving
@testset "Process LTI infeasible iter_max" begin
    Gmax = 10.0
    params2 = merge(params, (iter_max=5,))
    c_list, x_dx_list, deriv, flag = CLC.process_lyapunov_function(
        prob, x_list, G0, Gmax, r0, rmin, params2, solver)
    @test deriv > params2.tol_deriv
    @test !flag
end

end # TestMain