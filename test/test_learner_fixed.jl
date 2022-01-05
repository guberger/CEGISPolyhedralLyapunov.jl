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
n_piece = 5
method = CLC.LearnPolyhedralFixed{2}(n_piece)
A = [-0.3 0.0; -0.5 -0.3]
A_list = [A]
G0 = 0.1
r0 = 0.01
rmin = 1e-6
tol_faces = 1e-5
solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)
print_period = 1

N = 20
α_list = range(0.0, stop=2π, length=N)
x_list = map(α -> [cos(α), sin(α)], α_list)
x_dx_list = map(x -> (x, map(A -> A*x, A_list)), x_list)

## Solving
@testset "Learner LTI infeasible" begin
    Gmax = 0.1
    δ, c_list, G, r, flag = CLC.learn_candidate_lyapunov_function(
        method, x_dx_list,
        G0, Gmax, r0, rmin, tol_faces,
        print_period, solver)
    @test G == G0
    @test r == r0
    @test !flag
end

## Solving
@testset "Learner LTI feasible" begin
    Gmax = 10.0
    δ, c_list, G, r, flag = CLC.learn_candidate_lyapunov_function(
        method, x_dx_list,
        G0, Gmax, r0, rmin, tol_faces,
        print_period, solver)
    @test G == 0.2
    @test r == r0/2
    @test flag
    @test abs(δ - 0.061509706904423034) < 1e-9 # Gurobi sol, validated with plot
end

## Parameters
n_piece = 5
method = CLC.LearnPolyhedralFixed{2}(n_piece)
A1 = [-0.5 1.0; -1.0 -0.5]
A2 = [-0.3 0.0; -0.5 -0.3]
A_list = [A1, A2]
G0 = 0.1
r0 = 0.01
rmin = 1e-6
tol_faces = 1e-5
solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)
print_period = 1

N = 20
α_list = range(0.0, stop=2π, length=N)
x_list = map(α -> [cos(α), sin(α)], α_list)
x_dx_list = map(x -> (x, map(A -> A*x, A_list)), x_list)

## Solving
@testset "Learner SLS infeasible" begin
    Gmax = 1.0
    δ, c_list, G, r, flag = CLC.learn_candidate_lyapunov_function(
        method, x_dx_list,
        G0, Gmax, r0, rmin, tol_faces,
        print_period, solver)
    @test G == 0.8
    @test r == r0/8
    @test !flag
end

## Solving
@testset "Learner SLS feasible" begin
    Gmax = 10.0
    δ, c_list, G, r, flag = CLC.learn_candidate_lyapunov_function(
        method, x_dx_list,
        G0, Gmax, r0, rmin, tol_faces,
        print_period, solver)
    @test G == 3.2
    @test r == r0/32
    @test flag
    @test abs(δ - 0.054206489859162044) < 1e-9 # Gurobi sol, validated with plot
end

end # TestMain