using LinearAlgebra
using JuMP
using HiGHS
using CSDP
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

solverLP = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)
solverM = optimizer_with_attributes(CSDP.Optimizer, "printlevel"=>0)

## Parameters
ϵ = 0.5
tol = eps(1.0)
D = 2
M = 4
meth = CPL.MVE()

## Tests
domain = zeros(1, D)
fields = [[1.0 0; 0.0 1.0]]
sys = CPL.LinearSystem(domain, fields)
systems = (sys,)

seeds_init = (CPL.Node[],)

δ_min = 0.005
coeffs, nodes, obj_max, flag =
    CPL.process_PLF_fixed(meth, M, D, systems, seeds_init,
                          ϵ, tol, δ_min, solverLP, solverM,
                          depth_max=2,
                          level_output=2)

@testset "process_PLF_fixed MVE: infeasible" begin
    @test flag
    @test obj_max > tol
end

println("\nfinished-----------------------------------------------------------")