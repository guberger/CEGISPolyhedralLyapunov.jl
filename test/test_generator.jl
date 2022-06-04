using LinearAlgebra
using JuMP
using HiGHS
using Test
@static if isdefined(Main, :TestLocal)
    include("../src/CEGISPolyhedralVerification.jl")
else
    using CEGISPolyhedralVerification
end
CPV = CEGISPolyhedralVerification

solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)

## Parameters
ϵ = 1e5
θ = 1.0
δ = 1.0
nvar = 2
gen = CPV.Generator(nvar)

r = CPV.compute_lfs_feasibility(gen, ϵ, θ, δ, solver)[2]

@testset "compute lfs feasibility" begin
    @test r > 0
end

lfs, r = CPV.compute_lfs_chebyshev(gen, 1/θ, solver)

@testset "compute lfs chebyshev" begin
    @test r ≈ 2
    @test isempty(lfs)
end

A = [-1.0 0.0; 0.0 -1.0]
points = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]

for point in points
    wit = CPV.Witness()
    CPV.add_evidence_pos!(wit, point, norm(point, Inf))
    deriv = A*point
    CPV.add_evidence_lie!(wit, point, deriv, norm(point, Inf), 1.0, 1.0)
    CPV.add_witness!(gen, wit)
end

δ = 1.0 + 1e-5
r = CPV.compute_lfs_feasibility(gen, ϵ, θ, δ, solver)[2]

@testset "compute lfs feasibility" begin
    @test r < 0
end

# Gs = [0.25, 0.5, 1.0]

lfs, r = CPV.compute_lfs_chebyshev(gen, 1.0, solver)

@testset "compute lfs chebyshev" begin
    @test r ≈ 1/3
    @test maximum(lf -> norm(lf.lin, 1), lfs) ≈ 1
end

ϵ = 1e5
θ = 1.0
δ = 1e-5
Gs = [1.0]
nvar = 2
gen = CPV.Generator(nvar)

As = [
    [-1.0 0.0; 0.0 -2.0],
    [-2.0 0.0; 0.0 -1.0]
]

for point in points
    for A in As
        wit = CPV.Witness()
        CPV.add_evidence_pos!(wit, point, norm(point, Inf))
        deriv = A*point
        CPV.add_evidence_lie!(
            wit, point, deriv, norm(point, Inf), norm(point, Inf), 1.0
        )
        CPV.add_witness!(gen, wit)
    end
end

r = CPV.compute_lfs_feasibility(gen, ϵ, θ, δ, solver)[2]

@testset "compute lfs feasibility" begin
    @test r < 0
end

# Gs = [0.25, 0.5, 1.0, 2.0]

lfs, r = CPV.compute_lfs_chebyshev(gen, 2.0, solver)

@testset "compute lfs chebyshev" begin
    @test r ≈ 0.8/3
    @test maximum(lf -> norm(lf.lin, 1), lfs) ≈ 1
end

nothing