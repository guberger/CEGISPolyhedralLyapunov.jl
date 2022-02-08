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
n_piece = 3
meth_learn = CLC.LearnPolyhedralFixed{2}(n_piece)
meth_verify = CLC.VerifyPolyhedralMultiple{2}()
params = (tol_faces=1e-1, tol_deriv=eps(1.0),
          print_period_1=1, print_period_2=1, iter_max=100,
          do_trace=true)
solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)

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

x_list = [[-1.0, 0.0]]

@testset "Process Fixed: infeasible iter_max" begin
    G0 = Gmax = 1.0
    r0 = rmin = 0.0
    params2 = merge(params, (iter_max=1, do_trace=false))
    c_list, x_dx_list, deriv, flag, trace = CLC.process_lyapunov_function(
        prob, x_list, G0, Gmax, r0, rmin, params2, solver)
    @test length(c_list) == 6
    @test minimum(c -> abs(c[2]), c_list) < eps(100.0)
    @test deriv > params.tol_deriv
    @test !flag
    @test isempty(trace.c_list)
    @test isempty(trace.x_dx_list)
    @test isempty(trace.flag_learner)
    @test isempty(trace.x_dx)
    @test isempty(trace.flag_verifier)
end

x_list = [[-1.0, 0.0], [0.0, +1.0], [0.0, -1.0], [-1.0, +1.0], [-1.0, -1.0]]

@testset "Process Fixed: feasible" begin
    G0 = Gmax = 1.0
    r0 = rmin = 0.0
    c_list, x_dx_list, deriv, flag, trace = CLC.process_lyapunov_function(
        prob, x_list, G0, Gmax, r0, rmin, params, solver)
    @test minimum(c -> norm(c - [-1, 0]), c_list) < eps(100.0)
    @test abs(deriv) < eps(100.0)
    @test flag
    @test length(trace.c_list) == 2
    @test length(trace.x_dx_list) == 2
    @test trace.flag_learner == [true for i = 1:2]
    @test length(trace.x_dx) == 1
    @test trace.flag_verifier == [true for i = 1:2]
end

end # TestMain