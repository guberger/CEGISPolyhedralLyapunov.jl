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
nvar = 2
nloc = 1
gen = CPV.Generator(nvar, nloc)
_EYE_ = Matrix{Bool}(I, nvar, nvar)

ϵ = 1e5
δ = 1.0
r = CPV.compute_polyf_feasibility(gen, ϵ, δ, solver)[2]

@testset "compute lfs feasibility" begin
    @test r > 0
end

polyf, r = CPV.compute_polyf_chebyshev(gen, solver)

@testset "compute lfs chebyshev" begin
    @test r ≈ 2
    @test isempty(polyf.lfs)
    @test all(iset -> isempty(iset), polyf.loc_map)
end

A = [-1.0 0.0; 0.0 -1.0]
point1s = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]

τ = 1.0

for point1 in point1s
    wit = CPV.Witness()
    CPV.add_evidence_pos!(wit, point1, 1, norm(point1, Inf))
    point2 = point1 + τ*(A*point1)
    CPV.add_evidence_lie!(
        wit, point1, 1, point2, 1,
        norm(point1, Inf), norm(point2, Inf), norm(point2 - point1, Inf),
        opnorm(τ*A + _EYE_, Inf), opnorm(τ*A, Inf)
    )
    CPV.add_witness!(gen, wit)
end

ϵ = 1e5
δ = 1.0 + 1e-5
r = CPV.compute_polyf_feasibility(gen, ϵ, δ, solver)[2]

@testset "compute lfs feasibility" begin
    @test r < 0
end

polyf, r = CPV.compute_polyf_chebyshev(gen, solver)

@testset "compute lfs chebyshev" begin
    @test r ≈ 1
    @test maximum(lf -> norm(lf.lin, 1), polyf.lfs) ≈ 1
end

gen = CPV.Generator(nvar, nloc)

As = [
    [-1.0 0.0; 0.0 -2.0],
    [-2.0 0.0; 0.0 -1.0]
]

τ = 0.5

for point1 in point1s
    for A in As
        wit = CPV.Witness()
        CPV.add_evidence_pos!(wit, point1, 1, norm(point1, Inf))
        point2 = point1 + τ*(A*point1)
        CPV.add_evidence_lie!(
            wit, point1, 1, point2, 1,
            norm(point1, Inf), norm(point2, Inf), norm(point2 - point1, Inf),
            opnorm(τ*A + _EYE_, Inf), opnorm(τ*A, Inf)
        )
        CPV.add_witness!(gen, wit)
    end
end

ϵ = 1e5
δ = 1e-5
r = CPV.compute_polyf_feasibility(gen, ϵ, δ, solver)[2]

@testset "compute lfs feasibility" begin
    @test r > 0
end

polyf, r = CPV.compute_polyf_chebyshev(gen, solver)

@testset "compute lfs chebyshev" begin
    @test r ≈ 0.5
    @test maximum(lf -> norm(lf.lin, 1), polyf.lfs) ≈ 1
end

nvar = 2
nloc = 2
gen = CPV.Generator(nvar, nloc)

A_locs = [
    ([0.25 0.0; 0.0 0.25], 1, 2),
    ([2.0 0.0; 0.0 2.0], 2, 1)
]

for point1 in point1s
    for (A, loc1, loc2) in A_locs
        wit = CPV.Witness()
        CPV.add_evidence_pos!(wit, point1, loc1, norm(point1, Inf))
        point2 = A*point1
        CPV.add_evidence_lie!(
            wit, point1, loc1, point2, loc2,
            norm(point1, Inf), norm(point2, Inf), norm(point2 - point1, Inf),
            opnorm(A, Inf), opnorm(A - _EYE_, Inf)
        )
        CPV.add_witness!(gen, wit)
    end
end

ϵ = 1e5
δ = 1e-5
r = CPV.compute_polyf_feasibility(gen, ϵ, δ, solver)[2]

@testset "compute lfs feasibility" begin
    @test r > 0
end

polyf, r = CPV.compute_polyf_chebyshev(gen, solver)

@testset "compute lfs chebyshev" begin
    @test r ≈ 1/11
    @test maximum(lf -> norm(lf.lin, 1), polyf.lfs) ≈ 1
end

nothing