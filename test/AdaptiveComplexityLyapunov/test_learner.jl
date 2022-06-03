using LinearAlgebra
using JuMP
using HiGHS
using Test
@static if isdefined(Main, :TestLocal)
    include("../../src/CEGISPolyhedralVerification.jl")
else
    using CEGISPolyhedralVerification
end
CPLA = CEGISPolyhedralVerification.AdaptiveComplexityLyapunov
CPLP = CEGISPolyhedralVerification.Polyhedra

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

sys = CPLA.System()

domain = CPLP.Cone()
CPLP.add_supp!(domain, [1.0, 0.0])
A = [-1.0 1.0; -1.0 -1.0]
CPLA.add_piece!(sys, domain, A)

domain = CPLP.Cone()
CPLP.add_supp!(domain, [-1.0, 0.0])
A = [-1.0 -1.0; 1.0 -1.0]
CPLA.add_piece!(sys, domain, A)

lear = CPLA.Learner(nvar, sys, ϵ, θ, δ)
CPLA.set_tol!(lear, :rad, 0.0)
point = [-1.0, 0.0]
CPLA.add_point_init!(lear, point)

sol = CPLA.learn_lyapunov!(lear, 1, solver)

@testset "learn lyapunov: max iter" begin
    @test sol.status == CPLA.MAX_ITER_REACHED
end

sol = CPLA.learn_lyapunov!(lear, 100, solver)

@testset "learn lyapunov: infeasible" begin
    @test sol.status == CPLA.LYAPUNOV_INFEASIBLE
end

δ = 0.1
lear = CPLA.Learner(nvar, sys, ϵ, θ, δ)
CPLA.set_tol!(lear, :rad, 0.0)
point = [-1.0, 0.0]
CPLA.add_point_init!(lear, point)

sol = CPLA.learn_lyapunov!(lear, 100, solver, do_print=false)

@testset "learn lyapunov: feasible" begin
    @test sol.status == CPLA.LYAPUNOV_FOUND
end

CPLA.set_tol!(lear, :rad, 0.2)

sol = CPLA.learn_lyapunov!(lear, 100, solver, do_print=false)

@testset "learn lyapunov: radius too small" begin
    @test sol.status == CPLA.RADIUS_TOO_SMALL
end

nothing