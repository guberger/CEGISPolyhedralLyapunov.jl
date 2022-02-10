using StaticArrays
using LinearAlgebra
using JuMP
using Test
@static if isdefined(Main, :TestLocal)
    include("../src/CEGPolyhedralLyapunov.jl")
else
    using CEGPolyhedralLyapunov
end
CPL = CEGPolyhedralLyapunov

witness_svec = CPL.Witness(SVector{2}([1,0]), [SVector{2}([6,3])])
witness_vec = CPL.Witness(Val(2), [1,0], [[6,3]])

@testset "Witness" begin
    @inferred CPL.Witness(Val(2), [1,0], [[6,3]])
    @test true
    @inferred CPL.Witness(SVector{2}([1,0]), [SVector{2}([6,3])])
    @test true
    for witness in (witness_vec, witness_svec)
        @test CPL.state_dim(witness) == 2
        @test CPL.state_dim(typeof(witness)) == 2
    end
end

consts_vec = [m*ones(5) for m = 1:6]
fields_vec = [ones(5, 5), zeros(5, 5)]
sys_vec = CPL.LinearSystem(Val(5), consts_vec, fields_vec)
consts_svec = [m*ones(SVector{5,Int}) for m = 1:6]
fields_svec = [ones(SMatrix{5,5,Int}), zeros(SMatrix{5,5,Int})]
sys_svec = CPL.LinearSystem(consts_svec, fields_svec)

@testset "LinearSystem" begin
    @inferred CPL.LinearSystem(Val(5), consts_vec, fields_vec)
    @test true
    @inferred CPL.LinearSystem(consts_svec, fields_svec)
    @test true
    for sys in (sys_vec, sys_svec)
        @test CPL.state_dim(sys) == 5
        @test CPL.state_dim(typeof(sys)) == 5
    end
end

consts1 = [[+1, 0]]
fields1 = [[-1 0; 0 0], [-1 0.1; 0 -1]]
consts2 = [[-1, 0]]
fields2 = [[+1 0.1; 0 0], [0.5 0; 0 0.5]]
sys1 = CPL.LinearSystem(Val(2), consts1, fields1)
sys2 = CPL.LinearSystem(Val(2), consts2, fields2)
systems = (sys1, sys2)
points = ([0, 1], [1, 1])
witnesses = CPL.make_witnesses(systems, points)

@testset "make_witnesses" begin
    @test length(witnesses) == 3
    for witness in witnesses
        @test length(witness.flows) == 2
    end
end

hc = CPL.make_hypercube(Val(5))

@testset "make_hypercube" begin
    @inferred CPL.make_hypercube(Val(5))
    @test true
    for α in range(0, 2π, length=100)
        x = [cos(i*α) for i = 1:5]
        @test norm(x, Inf) ≈ maximum(h -> dot(h, x), hc)
    end
end

println("\nfinished-----------------------------------------------------------")