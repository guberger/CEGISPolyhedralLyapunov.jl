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
A = [-0.3 0.0; -0.5 -0.3]
G0 = 0.1
tol_faces = 1e-5
solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)
print_period = 1

N = 20
α_list = range(0.0, stop=2π, length=N)
x_list = map(α -> [cos(α), sin(α)], α_list)
x_dx_list = map(x -> (x, map(A -> A*x, [A, A])), x_list)

## Solving
@testset "Learner" begin
    Gmax = 0.1
    r, c_list, G, flag = CLC.learn_candidate_lyapunov_function(
        method, x_dx_list,
        G0, Gmax, tol_faces,
        print_period, solver)
    @test G == G0
    @test !flag
end

## Solving
@testset "Learner" begin
    Gmax = 10.0
    r, c_list, G, flag = CLC.learn_candidate_lyapunov_function(
        method, x_dx_list,
        G0, Gmax, tol_faces,
        print_period, solver)
    @test G == 0.2
    @test flag
    @test abs(r - 0.057104104253475306) < 1e-9 # Gurobi sol, validated with plot
end

## Parameters
method = CLC.LearnPolyhedralPoints{2}()
A1 = [-0.5 1.0; -1.0 -0.5]
A2 = [-0.3 0.0; -0.5 -0.3]
A_list = [A1, A2]
G0 = 0.1
tol_faces = 1e-5
solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)
print_period = 1

N = 20
α_list = range(0.0, stop=2π, length=N)
x_list = map(α -> [cos(α), sin(α)], α_list)
x_dx_list = map(x -> (x, map(A -> A*x, A_list)), x_list)

## Solving
@testset "Learner" begin
    Gmax = 1.0
    r, c_list, G, flag = CLC.learn_candidate_lyapunov_function(
        method, x_dx_list,
        G0, Gmax, tol_faces,
        print_period, solver)
    @test G == 0.8
    @test !flag
end

## Solving
@testset "Learner" begin
    Gmax = 10.0
    r, c_list, G, flag = CLC.learn_candidate_lyapunov_function(
        method, x_dx_list,
        G0, Gmax, tol_faces,
        print_period, solver)
    @test G == 1.6
    @test flag
    @test abs(r - 0.1056926539443272) < 1e-9 # Gurobi sol, validated with plot
end

end # TestMain