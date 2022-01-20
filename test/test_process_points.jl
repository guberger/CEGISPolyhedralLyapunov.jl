module TestMain

using LinearAlgebra
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
Hs1 = [[+1.0, 0.0], [0.0, -1.0]]
As1 = [[0.0 +1.0; -1.0 0.0]]
Hs2 = [[+1.0, 0.0], [0.0, +1.0]]
As2 = [[0.0 -1.0; +1.0 0.0]]
Hs3 = [[-1.0, 0.0]]
As3 = [[0.0 0.0; 0.0 -1.0]]
As_list = [As1, As2, As3]
Hs_list = [Hs1, Hs2, Hs3]
sys = CLC.PiecewiseLinearSystem{2}(3, Hs_list, As_list)
prob = CLC.CEGARProblem(sys, meth_learn, meth_verify)
params = (tol_faces=1e-1, tol_deriv=eps(1.0),
          print_period_1=1, print_period_2=1, iter_max=100,
          do_trace=true)
solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)

x_list = [[-1.0, 0.0], [0.0, +1.0], [0.0, -1.0]]

@testset "Process LTI infeasible iter_max" begin
    G0 = Gmax = 1.0
    r0 = rmin = 0.0
    params2 = merge(params, (iter_max=1, do_trace=false))
    c_list, x_dx_list, deriv, flag, trace = CLC.process_lyapunov_function(
        prob, x_list, G0, Gmax, r0, rmin, params2, solver)
    ϵ = params.tol_faces
    @test norm(c_list[1] - [-1, 0]) < eps(100.0)
    @test norm(c_list[2] - [-1, 0]) < eps(100.0)
    @test norm(c_list[3] - [0, +ϵ]) < eps(100.0)
    @test norm(c_list[4] - [0, +ϵ]) < eps(100.0)
    @test norm(c_list[5] - [0, -ϵ]) < eps(100.0)
    @test norm(c_list[6] - [0, -ϵ]) < eps(100.0)
    @test abs(deriv - ϵ) < eps(ϵ*100)
    @test !flag
    @test isempty(trace.c_list)
    @test isempty(trace.x_dx_list)
    @test isempty(trace.flag_learner)
    @test isempty(trace.x_dx)
    @test isempty(trace.flag_verifier)
end

@testset "Process LTI infeasible iter_max" begin
    G0 = Gmax = 1.0
    r0 = rmin = 0.0
    c_list, x_dx_list, deriv, flag, trace = CLC.process_lyapunov_function(
        prob, x_list, G0, Gmax, r0, rmin, params, solver)
    @test norm(c_list[1] - [-1, 0]) < eps(100.0)
    @test norm(c_list[2] - [-1, 0]) < eps(100.0)
    @test abs(deriv) < eps(100.0)
    @test flag
    @test length(trace.c_list) == 3
    @test length(trace.x_dx_list) == 3
    @test trace.flag_learner == [true for i = 1:3]
    @test length(trace.x_dx) == 2
    @test trace.flag_verifier == [true for i = 1:3]
end

end # TestMain