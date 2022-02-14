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
grads = Vector{Float64}[]
CPL.add_grads!(grads, systems, [0, 1])
flow1 = CPL.make_flow(systems, [0, 1])
flow2 = CPL.make_flow(systems, [1, 1])

@testset "add_grads! & make_flow" begin
    @test length(grads) == 4
    @test length(flow1.grads) == 4
    @test length(flow2.grads) == 2
end

hc = CPL.hypercube(5)

@testset "hypercube" begin
    for α in range(0, 2π, length=100)
        x = [cos(i*α) for i = 1:5]
        @test norm(x, Inf) ≈ maximum(h -> dot(h, x), hc)
    end
end

D = 2
domain = zeros(1, D)
fields = [[-1.0 0.0; 0.0 -1.0]]
sys = CPL.LinearSystem(domain, fields)

flow1 = CPL.make_flow((sys,), [-1, 0])
flow2 = CPL.make_flow((sys,), [+1, 0])
witness1 = CPL.Witness(flow1, 1)
witness2 = CPL.Witness(flow2, 2)
node1 = CPL.Node(witness1, 1)
node2 = CPL.Node(witness1, 2)
node3 = CPL.Node(witness2, 2)
nodes = (node1, node2, node3)
tree = CPL.seed(nodes)

@testset "Tree" begin
    @test length(tree) == 3
    i = 3
    tail = tree
    while length(tail) > 0
        @test length(tail) == i
        i -= 1
        tail = tail.tail
    end
    @test iszero(length(tail))
    for node in nodes
        flag = false
        for node_t in tree
            flag = flag || node == node_t
        end
        @test flag
    end
    for node_t in tree
        flag = false
        for node in nodes
            flag = flag || node_t == node
        end
        @test flag
    end
end

println("\nfinished-----------------------------------------------------------")