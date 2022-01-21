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
method = CLC.LearnPolyhedralPoints{2}()
tol_faces = 1e-5
solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)
print_period = 1

## Solving
@testset "Learner Points: empty x_dx_list" begin
    G0 = Gmax = 1.0
    r0 = rmin = 1.0 + 1e-5
    δ, c_list, G, r, flag = CLC.learn_candidate_lyapunov_function(
        method, CLC.x_dx_type[],
        G0, Gmax, r0, rmin, tol_faces,
        print_period, solver)
    @test isinf(δ)
    @test isempty(c_list)
    @test G == G0
    @test r == r0
    @test flag
end

A = [-1.0 0.0; 0.0 -1.0]
A_list = [A]

x_list = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
x_dx_list = map(x -> (x, map(A -> A*x, A_list)), x_list)

## Solving
@testset "Learner Points: LTI infeasible" begin
    G0 = Gmax = 1.0
    r0 = rmin = 1.0 + 1e-5
    δ, c_list, G, r, flag = CLC.learn_candidate_lyapunov_function(
        method, x_dx_list,
        G0, Gmax, r0, rmin, tol_faces,
        print_period, solver)
    @test G == G0
    @test r == r0
    @test !flag
end

## Solving
@testset "Learner Points: LTI feasible" begin
    G0 = 0.25
    Gmax = 1.0 + 1e-5
    r0 = 4.0 - 1e-5
    rmin = 0.0
    δ, c_list, G, r, flag = CLC.learn_candidate_lyapunov_function(
        method, x_dx_list,
        G0, Gmax, r0, rmin, tol_faces,
        print_period, solver)
    @test G == 1.0
    @test r == r0/4
    @test flag
    @test abs(δ - 1.0) < 1e-6
end

## Parameters
A1 = [-1.0 0.0; 0.0 -2.0]
A2 = [-2.0 0.0; 0.0 -1.0]
A_list = [A1, A2]

x_list = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
x_dx_list = map(x -> (x, map(A -> A*x, A_list)), x_list)

## Solving
@testset "Learner Points: SLS infeasible" begin
    G0 = Gmax = 1.0
    r0 = rmin = 0.0 + 1e-5
    δ, c_list, G, r, flag = CLC.learn_candidate_lyapunov_function(
        method, x_dx_list,
        G0, Gmax, r0, rmin, tol_faces,
        print_period, solver)
    @test G == G0
    @test r == r0
    @test !flag
end

## Solving
@testset "Learner Points: SLS feasible" begin
    G0 = 0.25
    Gmax = 2.0 + 1e-5
    r0 = 8.0 - 1e-5
    rmin = 0.0
    δ, c_list, G, r, flag = CLC.learn_candidate_lyapunov_function(
        method, x_dx_list,
        G0, Gmax, r0, rmin, tol_faces,
        print_period, solver)
    @test G == 2.0
    @test r == r0/8
    @test flag
    @test abs(δ - 1.0) < 1e-6
end

end # TestMain