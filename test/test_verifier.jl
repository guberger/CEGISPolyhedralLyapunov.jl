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
# method_s = CLC.VerifyPolyhedralSingle{2}()
method_m = CLC.VerifyPolyhedralMultiple{2}()
A = [0.0 0.0; 1.0 0.0]
A_list = [A]
tol_faces = 1e-5
solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)

N = 5
α_list = range(0.0, stop=2π, length=N)
c_list = map(α -> [cos(α), sin(α)], α_list)

## Solving
@testset "Verifier LTI feasible" begin
    obj_max, x = CLC.verify_candidate_lyapunov_function(
        method_m, A_list, c_list, tol_faces, solver)
    @test abs(obj_max - 1.0) < 1e-9
    @test norm(x - [1.0, 1.0]) < 1e-9
end

## Parameters
A = [-3.0 0.0; 1.0 -3.0]
A_list = [A]

## Solving
@testset "Verifier LTI infeasible" begin
    obj_max, x = CLC.verify_candidate_lyapunov_function(
        method_m, A_list, c_list, tol_faces, solver)
    @test abs(obj_max + 2.0) < 1e-9
    @test norm(x - [1.0, 1.0]) < 1e-9
end

## Parameters
A1 = [0.1 1.0; -1.0 0.1]
A2 = [0.1 0.0; -0.5 0.1]
A_list = [A1, A2]
tol_faces = 1e-5

## Solving
@testset "Verifier LTI feasible" begin
    obj_max, x = CLC.verify_candidate_lyapunov_function(
        method_m, A_list, c_list, tol_faces, solver)
    @test abs(obj_max - 1.1) < 1e-9
    @test norm(abs.(x) - [1.0, 1.0]) < 1e-9 # Gurobi sol, validated with plot
end

## Parameters
A1 = [-2.0 1.0; -1.0 -2.0]
A2 = [-2.0 0.0; -0.5 -2.0]
A_list = [A1, A2]

## Solving
@testset "Verifier LTI infeasible" begin
    obj_max, x = CLC.verify_candidate_lyapunov_function(
        method_m, A_list, c_list, tol_faces, solver)
    @test abs(obj_max + 1) < 1e-9
    @test norm(abs.(x) - [1.0, 1.0]) < 1e-9 # Gurobi sol, validated with plot
end

end # TestMain