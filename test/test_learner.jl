using LinearAlgebra
using JuMP
using HiGHS
using Test
@static if isdefined(Main, :TestLocal)
    include("../src/CEGISPolyhedralVerification.jl")
else
    using CEGISPolyhedralVerification
end
CPV = CEGISPolyhedralVerification
Cone = CPV.Cone
State = CPV.State
System = CPV.System

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
δ = 0.1
nvar = 2
nloc = 1
_EYE_ = Matrix{Bool}(I, nvar, nvar)

sys = System()
τ = 0.1

domain = Cone()
CPV.add_supp!(domain, [1.0, 0.0])
DA = [-1.0 1.0; -1.0 -1.0]
A = _EYE_ + τ*DA
CPV.add_piece!(sys, domain, 1, A, 1)

domain = Cone()
CPV.add_supp!(domain, [-1.0, 0.0])
DA = [-1.0 -1.0; 1.0 -1.0]
τ = 1.0
A = _EYE_ + τ*DA
CPV.add_piece!(sys, domain, 1, A, 1)

lear = CPV.Learner(nvar, nloc, sys, ϵ, δ)
CPV.set_tol!(lear, :rad, 0.0)
point = [-1.0, 0.0]
CPV.add_state_init!(lear, point, 1)

sol = CPV.learn_lyapunov!(lear, 1, solver)

@testset "learn lyapunov: max iter" begin
    @test sol.status == CPV.MAX_ITER_REACHED
end

sol = CPV.learn_lyapunov!(lear, 100, solver)

@testset "learn lyapunov: infeasible" begin
    @test sol.status == CPV.LYAPUNOV_INFEASIBLE
end

δ = 0.01
lear = CPV.Learner(nvar, nloc, sys, ϵ, δ)
CPV.set_tol!(lear, :rad, 0.0)
point = [-1.0, 0.0]
CPV.add_state_init!(lear, point, 1)

sol = CPV.learn_lyapunov!(lear, 100, solver, do_print=false)

@testset "learn lyapunov: feasible" begin
    @test sol.status == CPV.LYAPUNOV_FOUND
end

CPV.set_tol!(lear, :rad, 0.1)

sol = CPV.learn_lyapunov!(lear, 100, solver, do_print=false)

@testset "learn lyapunov: radius too small" begin
    @test sol.status == CPV.RADIUS_TOO_SMALL
end

nothing