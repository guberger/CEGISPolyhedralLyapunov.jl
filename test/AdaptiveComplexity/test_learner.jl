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
CPLS = CEGISPolyhedralLyapunov.Polyhedra

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

domain = CPLS.Cone()
CPLS.add_supp!(domain, CPLS.Supp([1.0, 0.0]))
A = [-1.0 1.0; -1.0 -1.0]
CPLA.add_system!(prob, domain, A)

domain = CPLS.Cone()
CPLS.add_supp!(domain, CPLS.Supp([-1.0, 0.0]))
A = [-1.0 -1.0; 1.0 -1.0]
CPLA.add_system!(prob, domain, A)

point = [-1.0, 0.0]
CPLA.add_point!(prob, point)

vecs, r, trace_out = CPLA.learn_lyapunov(prob, 100, solver)

@testset "learn lyapunov: infeasible" begin
    @test isempty(vecs)
end

prob.δ = 0.1

vecs, r, trace_out = CPLA.learn_lyapunov(prob, 100, solver)

@testset "learn lyapunov: infeasible" begin
    @test !isempty(vecs)
end

nothing