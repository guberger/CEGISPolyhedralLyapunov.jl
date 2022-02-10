using LinearAlgebra
using JuMP
using Test
@static if isdefined(Main, :TestLocal)
    include("../src/CEGPolyhedralLyapunov.jl")
else
    using CEGPolyhedralLyapunov
end
CPL = CEGPolyhedralLyapunov

witness = CPL.Witness([1, 0], [[6, 3]], 0)

@testset "Witness" begin
    @test witness.point == [1, 0]
    @test witness.flows == [[6, 3]]
    @test witness.index == 0
end

domain = ones(Int, 6, 5)
fields = [ones(Int, 5, 5), zeros(Int, 5, 5)]
sys = CPL.LinearSystem(domain, fields)

@testset "LinearSystem" begin
    @test sys.domain == ones(6, 5)
    @test sys.fields == [ones(5, 5), zeros(5, 5)]
end

domain1 = [+1 0]
fields1 = [[-1 0; 0 0], [-1 0.1; 0 -1]]
domain2 = [-1 0]
fields2 = [[+1 0.1; 0 0], [0.5 0; 0 0.5]]
sys1 = CPL.LinearSystem(domain1, fields1)
sys2 = CPL.LinearSystem(domain2, fields2)
systems = (sys1, sys2)
points = ([0, 1], [1, 1])
witnesses = CPL.make_witnesses(systems, points)

@testset "make_witnesses" begin
    @test length(witnesses) == 3
    for witness in witnesses
        @test length(witness.flows) == 2
    end
end

hc = CPL.make_hypercube(5)

@testset "make_hypercube" begin
    for α in range(0, 2π, length=100)
        x = [cos(i*α) for i = 1:5]
        @test norm(x, Inf) ≈ maximum(h -> dot(h, x), hc)
    end
end

println("\nfinished-----------------------------------------------------------")