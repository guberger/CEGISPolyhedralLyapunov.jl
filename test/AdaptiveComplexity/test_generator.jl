using LinearAlgebra
using JuMP
using HiGHS
using Test
@static if isdefined(Main, :TestLocal)
    include("../../src/CEGISPolyhedralLyapunov.jl")
else
    using CEGISPolyhedralLyapunov
end
CPLA = CEGISPolyhedralLyapunov.AdaptiveComplexity

solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)

## Parameters
ϵ = 1e5
θ = 1.0
δ = 1.0
Gs = [1.0]
nvar = 2
vecsgen = CPLA.VecsGenerator(nvar, ϵ, θ, δ, Gs)

flag = CPLA.check_feasibility(vecsgen, solver)

@testset "check feasibility" begin
    @test flag
end

vecs, r = CPLA.compute_vecs(vecsgen, solver)

@testset "compute vecs" begin
    @test r ≈ 2
    @test vecsgen.rs ≈ [2]
    @test isempty(vecs)
end

A = [-1.0 0.0; 0.0 -1.0]
points = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]

for point in points
    wit = CPLA.Witness(point)
    deriv = A*point
    CPLA.add_deriv!(wit, deriv)
    local i = CPLA.add_vec!(vecsgen)
    CPLA.add_witness!(vecsgen, i, wit)
end

vecsgen.δ = 1.0 + 1e-5

flag = CPLA.check_feasibility(vecsgen, solver)

@testset "check feasibility" begin
    @test !flag
end

Gs = [0.25, 0.5, 1.0]
vecsgen.Gs = Gs
vecsgen.rs = fill(Inf, length(Gs))

vecs, r = CPLA.compute_vecs(vecsgen, solver)

@testset "compute_vecs" begin
    @test r ≈ 1/3
    @test maximum(vec -> norm(vec, 1), vecs) ≈ 1
end

ϵ = 1e5
θ = 1.0
δ = 1e-5
Gs = [1.0]
nvar = 2
vecsgen = CPLA.VecsGenerator(nvar, ϵ, θ, δ, Gs)

As = [
    [-1.0 0.0; 0.0 -2.0],
    [-2.0 0.0; 0.0 -1.0]
]

for point in points
    for A in As
        wit = CPLA.Witness(point)
        deriv = A*point
        CPLA.add_deriv!(wit, deriv)
        local i = CPLA.add_vec!(vecsgen)
        CPLA.add_witness!(vecsgen, i, wit)
    end
end

flag = CPLA.check_feasibility(vecsgen, solver)

@testset "check feasibility" begin
    @test !flag
end

Gs = [0.25, 0.5, 1.0, 2.0]
vecsgen.Gs = Gs
vecsgen.rs = fill(Inf, length(Gs))

vecs, r = CPLA.compute_vecs(vecsgen, solver)

@testset "compute_vecs" begin
    @test r ≈ 0.8/3
    @test maximum(vec -> norm(vec, 1), vecs) ≈ 1
end

nothing