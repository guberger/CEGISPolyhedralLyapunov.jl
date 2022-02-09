module TestMain

using LinearAlgebra
using JuMP
using HiGHS
using Test
@static if isdefined(Main, :TestLocal)
    include("../src/CEGPolyhedralLyapunov.jl")
else
    using CEGPolyhedralLyapunov
end
CPL = CEGPolyhedralLyapunov

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
# method_s = CPL.VerifyPolyhedralSingle{2}()
method_m = CPL.VerifyPolyhedralMultiple{2}()
tol_faces = 1e-5
solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)

Hs1 = [[+1.0, 0.0]]
As1 = [[-1.0 0.0; 0.0 0.0], [-1.0 0.1; 0.0 -1.0]]
Hs2 = [[-1.0, 0.0]]
As2 = [[1.0 0.1; 0.0 0.0], [0.5 0.0; 0.0 0.5]]
As_list = [As1, As2]
Hs_list = [Hs1, Hs2]
sys = CPL.PiecewiseLinearSystem{2}(2, Hs_list, As_list)

c_list = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]

## Solving
@testset "Verifier: #1" begin
    obj_max, x, flag, j, q, qA = CPL.verify_candidate_lyapunov_function(
        method_m, sys, c_list, tol_faces, solver)
    @test abs(obj_max - 1.1) < 1e-9
    @test norm(x - [1.0, 1.0]) < 1e-9
    @test flag
    @test j == 2
    @test q == 2
    @test qA == 1
end

## Parameters
As_list[2] = [zeros(2, 2), zeros(2, 2), [0.0 0.0; 1.0 0.0]]

## Solving
@testset "Verifier: #2" begin
    obj_max, x, flag, j, q, qA = CPL.verify_candidate_lyapunov_function(
        method_m, sys, c_list, tol_faces, solver)
    @test abs(obj_max - 1.0) < 1e-9
    @test norm(x - [1.0, 1.0]) < 1e-9
    @test flag
    @test j == 4
    @test q == 2
    @test qA == 3
end

## Parameters
As_list[1] = [[-1.0 0.1; 0.0 -1.0]]
As_list[2] = [[-3.0 0.0; 0.0 -3.0]]

## Solving
@testset "Verifier: #3" begin
    obj_max, x, flag, j, q, qA = CPL.verify_candidate_lyapunov_function(
        method_m, sys, c_list, tol_faces, solver)
    @test abs(obj_max + 0.9) < 1e-9
    @test norm(x - [-1.0, -1.0]) < 1e-9
end

end # TestMain