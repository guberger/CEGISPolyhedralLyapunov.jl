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
nvar = 2
gen = CPLA.Generator(nvar)

r = CPLA.compute_vecs_feasibility(gen, ϵ, θ, δ, solver)[2]

@testset "compute vecs feasibility" begin
    @test r > 0
end

vecs, r = CPLA.compute_vecs_chebyshev(gen, 1/θ, solver)

@testset "compute vecs chebyshev" begin
    @test r ≈ 2
    @test isempty(vecs)
end

A = [-1.0 0.0; 0.0 -1.0]
points = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]

for point in points
    wit = CPLA.Witness()
    CPLA.add_evidence_pos!(wit, point, norm(point, Inf))
    deriv = A*point
    CPLA.add_evidence_lie!(wit, point, deriv, norm(point, Inf), 1.0, 1.0)
    CPLA.add_witness!(gen, wit)
end

δ = 1.0 + 1e-5
r = CPLA.compute_vecs_feasibility(gen, ϵ, θ, δ, solver)[2]

@testset "compute vecs feasibility" begin
    @test r < 0
end

# Gs = [0.25, 0.5, 1.0]

vecs, r = CPLA.compute_vecs_chebyshev(gen, 1.0, solver)

@testset "compute vecs chebyshev" begin
    @test r ≈ 1/3
    @test maximum(vec -> norm(vec, 1), vecs) ≈ 1
end

ϵ = 1e5
θ = 1.0
δ = 1e-5
Gs = [1.0]
nvar = 2
gen = CPLA.Generator(nvar)

As = [
    [-1.0 0.0; 0.0 -2.0],
    [-2.0 0.0; 0.0 -1.0]
]

for point in points
    for A in As
        wit = CPLA.Witness()
        CPLA.add_evidence_pos!(wit, point, norm(point, Inf))
        deriv = A*point
        CPLA.add_evidence_lie!(
            wit, point, deriv, norm(point, Inf), norm(point, Inf), 1.0
        )
        CPLA.add_witness!(gen, wit)
    end
end

r = CPLA.compute_vecs_feasibility(gen, ϵ, θ, δ, solver)[2]

@testset "compute vecs feasibility" begin
    @test r < 0
end

# Gs = [0.25, 0.5, 1.0, 2.0]

vecs, r = CPLA.compute_vecs_chebyshev(gen, 2.0, solver)

@testset "compute vecs chebyshev" begin
    @test r ≈ 0.8/3
    @test maximum(vec -> norm(vec, 1), vecs) ≈ 1
end

nothing