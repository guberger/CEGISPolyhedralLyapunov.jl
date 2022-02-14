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
D = 2

## Tests
domain = zeros(1, D)
fields = [[0 1; 0 0]]
sys = CPL.LinearSystem(domain, fields)
systems = (sys,)

coeffs = [[-1, 0], [1, 0]]
M = length(coeffs)
obj_max, x, flag, i, q, σ = CPL.verify_PLF(M, D, systems, coeffs, Inf, solver)

@testset "Verifier: unbounded" begin
    @test !flag
end

ζ = 100
obj_max, x, flag, i, q, σ = CPL.verify_PLF(M, D, systems, coeffs, ζ, solver)

@testset "Verifier: reach bound" begin
    @test abs(obj_max - ζ/norm([1, ζ])) < 1e-7
    @test flag
end

domain1 = [+1 0]
fields1 = [[-1 0; 0 0], [-1 0.1; 0 -1]]
domain2 = [-1 0]
fields2 = [[+1 0.1; 0 0], [0.5 0; 0 0.5]]
sys1 = CPL.LinearSystem(domain1, fields1)
sys2 = CPL.LinearSystem(domain2, fields2)
systems = (sys1, sys2)

coeffs = [[-1, 0], [1, 0], [0, -1], [0, 1]]
M = length(coeffs)
obj_max, x, flag, i, q, σ = CPL.verify_PLF(M, D, systems, coeffs, Inf, solver)

@testset "Verifier: #1" begin
    @test abs(obj_max - 1.1/sqrt(2)) < 1e-9
    @test norm(x - [1, 1]) < 1e-9
    @test flag
    @test i == 2
    @test q == 2
    @test σ == 1
end

fields2 = [zeros(2, 2), zeros(2, 2), [0 0; 1 0]]
sys2 = CPL.LinearSystem(domain2, fields2)
systems = (sys1, sys2)

obj_max, x, flag, i, q, σ = CPL.verify_PLF(M, D, systems, coeffs, Inf, solver)

@testset "Verifier: #2" begin
    @test abs(obj_max - 1/sqrt(2)) < 1e-9
    @test norm(x - [1, 1]) < 1e-9
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

obj_max, x, flag, i, q, σ = CPL.verify_PLF(M, D, systems, coeffs, Inf, solver)

@testset "Verifier: #3" begin
    @test abs(obj_max + 0.9/sqrt(2)) < 1e-9
    @test norm(x - [-1, -1]) < 1e-9
end

domain = zeros(1, D)
fields = [[-101.1 99; 101 -99.1], [-101.1 -99; -101 -99.1]]
sys = CPL.LinearSystem(domain, fields)
systems = (sys,)

coeffs1 = [[-1, 0], [1, 0], [0, -1], [0, 1]]
coeffs2 = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
M = length(coeffs)
obj_max1 = CPL.verify_PLF(M, D, systems, coeffs1, Inf, solver)[1]
obj_max2 = CPL.verify_PLF(M, D, systems, coeffs2, Inf, solver)[1]

@testset "Verifier: #4" begin
    @test abs(obj_max1 - 1.9/sqrt(2)) < 1e-7
    @test abs(obj_max2 + 0.1) < 1e-7
end

println("\nfinished-----------------------------------------------------------")