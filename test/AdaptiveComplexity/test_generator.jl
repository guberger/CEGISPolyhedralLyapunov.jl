using LinearAlgebra
using JuMP
using HiGHS
using Test
@static if isdefined(Main, :TestLocal)
    include("../../src/CEGISPolyhedralLyapunov.jl")
else
    using CEGISPolyhedralLyapunov
end
CPLG = CEGISPolyhedralLyapunov.AdaptiveComplexity.Generator

solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)

## Parameters
ϵ = 1e5
θ = 1.0
δ = 1.0
nvar = 2
gen = CPLG.GeneratingProblem(nvar)

r = CPLG.compute_feasibility(gen, ϵ, θ, δ, solver)[2]

@testset "compute feasibility" begin
    @test r > 0
end

vecs, r = CPLG.compute_vecs_r_heuristic(gen, 1/θ, solver)

@testset "compute vecs" begin
    @test r ≈ 2
    @test isempty(vecs)
end

A = [-1.0 0.0; 0.0 -1.0]
points = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]

for point in points
    wit = CPLG.Witness()
    CPLG.add_evidence!(wit, CPLG.PosEvidence(point, norm(point, Inf)))
    deriv = A*point
    CPLG.add_evidence!(
        wit,
        CPLG.LieEvidence(point, deriv, norm(point, Inf), 1.0, 1.0
    ))
    CPLG.add_witness!(gen, wit)
end

δ = 1.0 + 1e-5
r = CPLG.compute_feasibility(gen, ϵ, θ, δ, solver)[2]

@testset "compute feasibility" begin
    @test r < 0
end

# Gs = [0.25, 0.5, 1.0]

vecs, r = CPLG.compute_vecs_r_heuristic(gen, 1.0, solver)

@testset "compute_vecs" begin
    @test r ≈ 1/3
    @test maximum(vec -> norm(vec, 1), vecs) ≈ 1
end

ϵ = 1e5
θ = 1.0
δ = 1e-5
Gs = [1.0]
nvar = 2
gen = CPLG.GeneratingProblem(nvar)

As = [
    [-1.0 0.0; 0.0 -2.0],
    [-2.0 0.0; 0.0 -1.0]
]

for point in points
    for A in As
        wit = CPLG.Witness()
        CPLG.add_evidence!(wit, CPLG.PosEvidence(point, norm(point, Inf)))
        deriv = A*point
        CPLG.add_evidence!(
            wit,
            CPLG.LieEvidence(point, deriv, norm(point, Inf), norm(point, Inf), 1.0
        ))
        CPLG.add_witness!(gen, wit)
    end
end

r = CPLG.compute_feasibility(gen, ϵ, θ, δ, solver)[2]

@testset "compute feasibility" begin
    @test r < 0
end

# Gs = [0.25, 0.5, 1.0, 2.0]

vecs, r = CPLG.compute_vecs_r_heuristic(gen, 2.0, solver)

@testset "compute_vecs" begin
    @test r ≈ 0.8/3
    @test maximum(vec -> norm(vec, 1), vecs) ≈ 1
end

nothing