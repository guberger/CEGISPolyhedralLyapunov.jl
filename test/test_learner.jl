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

τ = 1/4 # 0.1
tol_r = 0.05 # 0.1

pieces = [
    CPL.Piece([-0.5 1.0; -1.0 -0.5], [[0.0, -1.0]]),
    CPL.Piece([0.01 1.0; -1.0 0.01], [[0.0, 1.0]])
]

lfs_init = [[0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1]]

xmax = 15
iter_max = 2

status, lfs = CPL.learn_lyapunov(
    pieces, lfs_init, τ, 2, xmax, iter_max, solver, tol_r=tol_r
)

@testset "iter max" begin
    @test status == CPL.MAX_ITER_REACHED
end

τ = 1/4 # 0.1
tol_r = 0.05 # 0.1

pieces = [
    CPL.Piece([-0.5 1.0; -1.0 -0.5], [[0.0, -1.0]]),
    CPL.Piece([0.01 1.0; -1.0 0.01], [[0.0, 1.0]])
]

lfs_init = [[0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1]]

xmax = 15
iter_max = 10

status, lfs = CPL.learn_lyapunov(
    pieces, lfs_init, τ, 2, xmax, iter_max, solver, tol_r=tol_r
)

@testset "infeasible" begin
    @test status == CPL.LYAPUNOV_INFEASIBLE
end

τ = 1/4 # 0.1
tol_r = 0.001 # 0.1

pieces = [
    CPL.Piece([-0.5 1.0; -1.0 -0.5], [[0.0, -1.0]]),
    CPL.Piece([0.01 1.0; -1.0 0.01], [[0.0, 1.0]])
]

lfs_init = [[0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1]]

xmax = 15
iter_max = 50

status, lfs = CPL.learn_lyapunov(
    pieces, lfs_init, τ, 2, xmax, iter_max, solver, tol_r=tol_r
)

@testset "found" begin
    @test status == CPL.LYAPUNOV_FOUND
end

nothing