using LinearAlgebra
using JuMP
using HiGHS
using Test
@static if isdefined(Main, :TestLocal)
    include("../src/CEGISPolyhedralLyapunov.jl")
else
    using CEGISPolyhedralLyapunov
end
CPL = CEGISPolyhedralLyapunov

solver() = Model(optimizer_with_attributes(
    HiGHS.Optimizer, "output_flag"=>false
))

WT_ = CPL.Witness{Vector{Int},Float64,Vector{Vector{Int}}}
wit_cls = Vector{WT_}[]
lfs_init = Vector{Int}[]
rmax = 100

@testset "compute lfs empty" begin
    lfs, r = CPL.compute_lfs(wit_cls, lfs_init, 1, rmax, solver)
    @test isempty(lfs)
    @test r ≈ rmax
end

wit_cls = [[CPL.Witness([1], 3, [0.5])]]
lfs_init = Vector{Int}[]
rmax = 100

@testset "compute pf loop" begin
    lfs, r = CPL.compute_lfs(wit_cls, lfs_init, 1, rmax, solver)
    @test length(lfs) == 1
    @test lfs[1] ≈ [1]
    @test r ≈ 0.5/3
end

wit_cls = [[CPL.Witness([1], 3, [-1])]]
lfs_init = Vector{Int}[]
rmax = 100

@testset "compute pf no loop" begin
    lfs, r = CPL.compute_lfs(wit_cls, lfs_init, 1, rmax, solver)
    @test length(lfs) == 1
    @test lfs[1] ≈ [1]
    @test r ≈ 2/3
end

wit_cls = [[CPL.Witness([1], 3, [-0.5])]]
lfs_init = [[-0.25], [0.25]]
rmax = 100

@testset "compute pf init active" begin
    lfs, r = CPL.compute_lfs(wit_cls, lfs_init, 1, rmax, solver)
    @test length(lfs) == 1
    @test lfs[1] ≈ [1]
    @test r ≈ (1 - 0.25*0.5)/3
end

wit_cls = [
    [CPL.Witness([1], 3, [-1.0])],
    [CPL.Witness([-1], 3, [-2.0])],
]
lfs_init = [[-0.1], [0.1]]
rmax = 100

@testset "compute pf loop" begin
    lfs, r = CPL.compute_lfs(wit_cls, lfs_init, 1, rmax, solver)
    @test length(lfs) == 2
    @test lfs[1] ≈ [1]
    @test lfs[2] ≈ [-0.1]
    @test r ≈ -0.1/3
end

wit_cls = [
    [CPL.Witness([1], 3, [-1.0])],
    [CPL.Witness([-1], 3, [0.5])],
]
lfs_init = [[-0.1], [0.1]]
rmax = 100

@testset "compute pf cycle" begin
    lfs, r = CPL.compute_lfs(wit_cls, lfs_init, 1, rmax, solver)
    @test length(lfs) == 2
    @test lfs[1] ≈ [1]
    @test lfs[2] ≈ [-0.75]
    @test r ≈ 0.25/3
end

nothing