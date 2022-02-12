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

solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)

## Parameters
ϵ = 1e-1
tol = eps(1.0)
D = 2

## Tests
domain1 = [+1 0; 0 -1]
fields1 = [[0 +1; -1 0.01]]
domain2 = [+1 0; 0 +1]
fields2 = [[0 -1; +1 0.01]]
domain3 = [-1 0]
fields3 = [[0 0; 0 -1]]
sys1 = CPL.LinearSystem(domain1, fields1)
sys2 = CPL.LinearSystem(domain2, fields2)
sys3 = CPL.LinearSystem(domain3, fields3)
systems = (sys1, sys2, sys3)

x_init = [-1, 0]
flows_init = CPL.Flow[]
for sys in systems
    local flow = CPL.make_flow((sys,), x_init)
    isempty(flow.grads) && continue
    push!(flows_init, flow)
end
G0 = Gmax = 1.0
r0 = rmin = 0.0
coeffs, flows, deriv, flag, trace =
    CPL.process_PLF_adaptive(D, systems, flows_init,
                             G0, Gmax, r0, rmin, ϵ, tol,
                             solver, trace=false, iter_max=1)

@testset "Process Adaptive: infeasible iter_max" begin
    @test norm(coeffs[1] - [-1, 0]) < eps(100.0)
    @test norm(coeffs[2] - [-1, 0]) < eps(100.0)
    @test deriv > tol
    @test !flag
    @test isempty(trace.coeffs_list)
    @test isempty(trace.flows_list)
    @test isempty(trace.flags_learner)
    @test isempty(trace.counterexample_list)
    @test isempty(trace.flags_verifier)
end

coeffs, flows, deriv, flag, trace =
    CPL.process_PLF_adaptive(D, systems, flows_init,
                    G0, Gmax, r0, rmin, ϵ, tol,
                    solver, iter_max=100, learner_output=false)

@testset "Process Adaptive: feasible" begin
    @test norm(coeffs[1] - [-1, 0]) < eps(100.0)
    @test norm(coeffs[2] - [-1, 0]) < eps(100.0)
    @test abs(deriv) < eps(100.0)
    @test flag
    @test !isempty(trace.coeffs_list)
    @test !isempty(trace.flows_list)
    @test all(trace.flags_learner)
    @test !isempty(trace.counterexample_list)
    @test all(trace.flags_verifier)
end

println("\nfinished-----------------------------------------------------------")