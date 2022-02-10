using LinearAlgebra
using Test
@static if isdefined(Main, :TestLocal)
    include("../src/CEGPolyhedralLyapunov.jl")
else
    using CEGPolyhedralLyapunov
end
CPL = CEGPolyhedralLyapunov

flow = CPL.Flow([1, 0], [[6, 3]])

@testset "Flow" begin
    @test flow.point == [1, 0]
    @test flow.grads == [[6, 3]]
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
flows = CPL.make_flows(systems, points)

@testset "make_flows" begin
    @test length(flows) == 3
    for flow in flows
        @test length(flow.grads) == 2
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