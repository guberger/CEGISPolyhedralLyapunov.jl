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

A = [0 0; 0 0]

lf_max = [1, 1]
lfs_other = [[2, 0]]
lfs_dom = [[0, 1]]

xmax = 1e3

@testset "infeasible verify single" begin
    x, γ, flag = CPL.verify_single(
        A, lf_max, lfs_other, lfs_dom, 2, xmax, solver
    )
    @test all(isnan, x)
    @test γ ≈ -Inf
    @test !flag
end

pieces = CPL.Piece{Matrix{Int},Vector{Vector{Int}}}[]

lfs = [[1, 1], [1, -1], [-1, 1], [-1, -1]]

xmax = 1e3

@testset "empty pieces" begin
    x, γ, q, flag = CPL.verify(pieces, lfs, 2, xmax, solver)
    @test all(isnan, x)
    @test γ ≈ -Inf
    @test q == 0
    @test !flag
end

pieces = [
    CPL.Piece([0 1; 0 0], [[-1, 0]]),
    CPL.Piece([1 0; 0 0], [[1, 0]])
]

lfs = [[0, 0], [0, 0]]

xmax = 100

@testset "infeasible" begin
    x, γ, q, flag = CPL.verify(pieces, lfs, 2, xmax, solver)
    @test all(isnan, x)
    @test γ ≈ -Inf
    @test q == 0
    @test !flag
end

pieces = [
    CPL.Piece([0 1; 0 0], [[-1, 0]]),
    CPL.Piece([1 0; 0 0], [[1, 0]])
]

lfs = [[1, 0], [0, -1]]

xmax = 100

@testset "xmax reached" begin
    x, γ, q, flag = CPL.verify(pieces, lfs, 2, xmax, solver)
    @test x[2] ≈ xmax
    @test γ ≈ xmax
    @test q == 1
    @test flag
end

pieces = [
    CPL.Piece([0 -1; 0 0], [[-1, 0]]),
    CPL.Piece([1 0; 0 0], [[1, 0]])
]

lfs = [[1, 0], [0, -1], [0, 1]]

xmax = 100

@testset "optimal positive" begin
    x, γ, q, flag = CPL.verify(pieces, lfs, 2, xmax, solver)
    @test x ≈ [1, -1]
    @test γ ≈ 1
    @test q == 1
    @test flag
end

pieces = [
    CPL.Piece([-1 0; 0 0], [[-1, 0]]),
    CPL.Piece([1 0; 0 0], [[1, 0]])
]

lfs = [[1, 0], [0, -1], [0, 1]]

xmax = 100

@testset "optimal zero" begin
    x, γ, q, flag = CPL.verify(pieces, lfs, 2, xmax, solver)
    @test abs(x[2]) ≈ 1
    @test abs(γ) < 1e-6
    @test flag
end

pieces = [
    CPL.Piece([-1 0; 0 -0.5], [[-1, 0]]),
    CPL.Piece([1 0; 0 -0.1], [[1, 0]])
]

lfs = [[1, 0], [0, -1], [0, 1]]

xmax = 100

@testset "optimal negative" begin
    x, γ, q, flag = CPL.verify(pieces, lfs, 2, xmax, solver)
    @test abs(x[2]) ≈ 1
    @test γ ≈ -0.1
    @test q == 2
    @test flag
end

nothing