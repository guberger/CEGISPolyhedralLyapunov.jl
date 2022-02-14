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
# function HiGHS._check_ret(ret::Cint) 
#     if ret != Cint(0) && ret != Cint(1)
#         error(
#             "Encountered an error in HiGHS (Status $(ret)). Check the log " * 
#             "for details.", 
#         )
#     end 
#     return 
# end

solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)

## Parameters
ϵ = 1e-1
tol = eps(1.0)
D = 2
M = 4

## Tests
domain = zeros(1, D)
fields = [[1.0 0; 0.0 1.0]]
sys = CPL.LinearSystem(domain, fields)
systems = (sys,)

nodes_init = CPL.Node[]

δ_min = 0.45
coeffs, nodes, obj_max, flag =
    CPL.process_PLF_fixed(M, D, systems, nodes_init,
                          ϵ, tol, δ_min, solver,
                          iter_max=Inf,
                          output_period=100,
                          learner_output=false)

@testset "process_PLF_fixed: infeasible" begin
    @test flag
    @test obj_max > tol
end

fields = [[-101.1 99; 101 -99.1], [-101.1 -99; -101 -99.1]]
sys = CPL.LinearSystem(domain, fields)
systems = (sys,)

nodes_init = CPL.Node[]

δ_min = 0.25
coeffs, nodes, obj_max, flag =
    CPL.process_PLF_fixed(M, D, systems, nodes_init,
                          ϵ, tol, δ_min, solver,
                          iter_max=1000,
                          output_period=100,
                          learner_output=false)

@testset "process_PLF_fixed: feasible" begin
    @test flag
    @test abs(obj_max + 0.1) < eps(100.0)
    for s1 in (-1, +1), s2 in (-1, +1)
        @test [s1, s2] in coeffs
    end
end

println("\nfinished-----------------------------------------------------------")