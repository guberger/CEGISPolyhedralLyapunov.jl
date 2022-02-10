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
ϵ = 1e-5
D = 2

## Tests
domain1 = [+1 0]
fields1 = [[-1 0; 0 0], [-1 0.1; 0 -1]]
domain2 = [-1 0]
fields2 = [[+1 0.1; 0 0], [0.5 0; 0 0.5]]
sys1 = CPL.LinearSystem(domain1, fields1)
sys2 = CPL.LinearSystem(domain2, fields2)
systems = (sys1, sys2)

coeffs = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
obj_max, x, flag, i, q, σ = CPL.verify_PLF(D, systems, coeffs, ϵ, solver)

@testset "Verifier: #1" begin
    @test abs(obj_max - 1.1) < 1e-9
    @test norm(x - [1.0, 1.0]) < 1e-9
    @test flag
    @test i == 2
    @test q == 2
    @test σ == 1
end

fields2 = [zeros(2, 2), zeros(2, 2), [0.0 0.0; 1.0 0.0]]
sys2 = CPL.LinearSystem(domain2, fields2)
systems = (sys1, sys2)

obj_max, x, flag, i, q, σ = CPL.verify_PLF(D, systems, coeffs, ϵ, solver)

@testset "Verifier: #2" begin
    @test abs(obj_max - 1.0) < 1e-9
    @test norm(x - [1.0, 1.0]) < 1e-9
    @test flag
    @test i == 4
    @test q == 2
    @test σ == 3
end

fields1 = [[-1.0 0.1; 0.0 -1.0]]
fields2 = [[-3.0 0.0; 0.0 -3.0]]
sys1 = CPL.LinearSystem(domain1, fields1)
sys2 = CPL.LinearSystem(domain2, fields2)
systems = (sys1, sys2)

obj_max, x, flag, i, q, σ = CPL.verify_PLF(D, systems, coeffs, ϵ, solver)

@testset "Verifier: #3" begin
    @test abs(obj_max + 0.9) < 1e-9
    @test norm(x - [-1.0, -1.0]) < 1e-9
end

println("\nfinished-----------------------------------------------------------")