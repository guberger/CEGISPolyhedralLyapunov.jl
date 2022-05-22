using LinearAlgebra
using JuMP
using HiGHS
using Test
@static if isdefined(Main, :TestLocal)
    include("../../src/CEGISPolyhedralLyapunov.jl")
else
    using CEGISPolyhedralLyapunov
end
CPLA = CEGISPolyhedralLyapunov.AdaptiveComplexity
CPLP = CEGISPolyhedralLyapunov.Polyhedra

function HiGHS._check_ret(ret::Cint) 
    if ret != Cint(0) && ret != Cint(1)
        error(
            "Encountered an error in HiGHS (Status $(ret)). Check the log " * 
            "for details.", 
        )
    end 
    return 
end 

solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)

## Parameters
ϵ = 10.0
θ = 1.0
δ = 0.5
nvar = 2
prob = CPLA.LearningProblem(nvar, ϵ, θ, δ)
CPLA.set_tol_rad!(prob, 0.0)

α = 1.5
CPLA.set_Gs!(prob, α)

@testset "set Gs" begin
    @test prob.Gs ≈ [0.25*1.5^k for k = 0:4]
end

domain = CPLP.Cone()
CPLP.add_supp!(domain, CPLP.Supp([1.0, 0.0]))
A = [-1.0 1.0; -1.0 -1.0]
CPLA.add_system!(prob, domain, A)

domain = CPLP.Cone()
CPLP.add_supp!(domain, CPLP.Supp([-1.0, 0.0]))
A = [-1.0 -1.0; 1.0 -1.0]
CPLA.add_system!(prob, domain, A)

point = [-1.0, 0.0]
CPLA.add_point_init!(prob, point)

flag = CPLA.learn_lyapunov!(prob, 1, solver)

@testset "learn lyapunov: max iter" begin
    @test !flag
    @test prob.status == CPLA.MAX_ITER_REACHED
end

flag = CPLA.learn_lyapunov!(prob, 100, solver)

@testset "learn lyapunov: infeasible" begin
    @test !flag
    @test prob.status == CPLA.LYAPUNOV_INFEASIBLE
end

prob.δ = 0.1

flag = CPLA.learn_lyapunov!(prob, 100, solver)

@testset "learn lyapunov: feasible" begin
    @test flag
    @test prob.status == CPLA.LYAPUNOV_FOUND
end

CPLA.set_tol_rad!(prob, 0.2)
flag = CPLA.learn_lyapunov!(prob, 100, solver)

@testset "learn lyapunov: radius too small" begin
    @test !flag
    @test prob.status == CPLA.RADIUS_TOO_SMALL
end

nothing