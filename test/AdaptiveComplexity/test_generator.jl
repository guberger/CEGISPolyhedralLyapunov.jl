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
vecsgen = CPLA.VecsGenerator(nvar, Gs)

r = CPLA.compute_feasibility(vecsgen, ϵ, θ, solver)

@testset "compute feasibility" begin
    @test r > δ
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
    deriv = A*point
    pos_constrs = [CPLA.PosConstraint(point, norm(point, Inf))]
    lie_constrs = [CPLA.LieConstraint(
        point, deriv, norm(point, Inf), 1.0, 1.0
    )]
    CPLA.add_witness!(vecsgen, CPLA.Witness(pos_constrs, lie_constrs))
end

δ = 1.0 + 1e-5
r = CPLA.compute_feasibility(vecsgen, ϵ, θ, solver)

@testset "compute feasibility" begin
    @test r < δ
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
vecsgen = CPLA.VecsGenerator(nvar, Gs)

As = [
    [-1.0 0.0; 0.0 -2.0],
    [-2.0 0.0; 0.0 -1.0]
]

for point in points
    for A in As
        deriv = A*point
        pos_constrs = [CPLA.PosConstraint(point, norm(point, Inf))]
        lie_constrs = [CPLA.LieConstraint(
            point, deriv, norm(point, Inf), norm(point, Inf), 1.0
        )]
        CPLA.add_witness!(vecsgen, CPLA.Witness(pos_constrs, lie_constrs))
    end
end

r = CPLA.compute_feasibility(vecsgen, ϵ, θ, solver)

@testset "compute feasibility" begin
    @test r < δ
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