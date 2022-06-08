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
PosEvidence = CPV.PosEvidence
LieDiscEvidence = CPV.LieDiscEvidence
LieContEvidence = CPV.LieContEvidence
PolyFunc = CPV.PolyFunc
_norm(pf::PolyFunc) = maximum(lf -> norm(lf.lin, 1), pf.lfs)

solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)

## Parameters
nvar = 2

## Empty
nloc = 1
gen = CPV.Generator(nvar, nloc)

rf = CPV.compute_mpf_feasibility(gen, 1e5, 1.0, solver)[2]
mpf, rc = CPV.compute_mpf_chebyshev(gen, solver)
re = CPV.compute_mpf_evidence(gen, solver)[2]

@testset "compute mpf empty" begin
    @test rf > 0
    @test rc ≈ 2
    @test all(pf -> isempty(pf.lfs), mpf.pfs)
    @test re ≈ 2
end

## Pos
nloc = 1
gen = CPV.Generator(nvar, nloc)

CPV.add_lf!(gen, 1)
point = [2, 0]
CPV.add_evidence!(gen, PosEvidence(1, 1, point, norm(point, Inf)))

ϵ = 1e5
δ = 1.0
rf = CPV.compute_mpf_feasibility(gen, ϵ, δ, solver)[2]
mpf, rc = CPV.compute_mpf_chebyshev(gen, solver)
re = CPV.compute_mpf_evidence(gen, solver)[2]

@testset "compute mpf pos" begin
    @test rf ≈ δ - 1/ϵ
    @test rc ≈ 1
    @test maximum(pf -> _norm(pf), mpf.pfs) ≈ 1
    @test re ≈ 1
end

## Lie
nloc = 1
gen = CPV.Generator(nvar, nloc)

CPV.add_lf!(gen, 1)
point1 = [2, 0]
point2 = [4, 0]
τ = 2.0
CPV.add_evidence!(
    gen, LieContEvidence(
        1, 1, point1, point2,
        norm(point1, Inf), norm(point2, Inf),
        norm(point2 - point1, Inf), 1.0, 0.0, τ
    )
)

ϵ = 1e5
δ = 1.0
rf = CPV.compute_mpf_feasibility(gen, ϵ, δ, solver)[2]
mpf, rc = CPV.compute_mpf_chebyshev(gen, solver)
re = CPV.compute_mpf_evidence(gen, solver)[2]

@testset "compute mpf lie" begin
    @test rf ≈ 1 - δ*τ
    @test rc ≈ 1
    @test maximum(pf -> _norm(pf), mpf.pfs) ≈ 1
    @test re ≈ 2
end

## Pos and Lie: 1 wit
nloc = 1
gen = CPV.Generator(nvar, nloc)

CPV.add_lf!(gen, 1)
point1 = [2, 0]
point2 = [4, 0]
CPV.add_evidence!(gen, PosEvidence(1, 1, point1, norm(point1, Inf)))
CPV.add_evidence!(
    gen, LieContEvidence(
        1, 1, point1, point2,
        norm(point1, Inf), norm(point2, Inf),
        norm(point2 - point1, Inf), 1.0, 0.0, 1.0
    )
)

ϵ = 1e5
δ = 1.0
rf = CPV.compute_mpf_feasibility(gen, ϵ, δ, solver)[2]
mpf, rc = CPV.compute_mpf_chebyshev(gen, solver)
re = CPV.compute_mpf_evidence(gen, solver)[2]

@testset "compute mpf pos and lie: 1 wit" begin
    @test rf ≈ -(δ + 1/ϵ)/2
    @test rc ≈ 0
    @test maximum(pf -> _norm(pf), mpf.pfs) ≈ 0
    @test re ≈ 0
end

## Pos and Lie: 2 wits #1
nloc = 2
gen = CPV.Generator(nvar, nloc)

CPV.add_lf!(gen, 1)
CPV.add_lf!(gen, 2)
point = [2, 0]
CPV.add_evidence!(gen, PosEvidence(2, 1, point, norm(point, Inf)))
point1 = [2, 0]
point2 = [4, 0]
CPV.add_evidence!(
    gen, LieDiscEvidence(
        1, 1, point1, 2, point2,
        norm(point1, Inf), norm(point2, Inf),
        norm(point2 - point1, Inf), 2.0, 1.0
    )
)

ϵ = 1e5
δ = 1.0
rf = CPV.compute_mpf_feasibility(gen, ϵ, δ, solver)[2]
mpf, rc = CPV.compute_mpf_chebyshev(gen, solver)
re = CPV.compute_mpf_evidence(gen, solver)[2]

@testset "compute mpf pos and lie: 2 wits #1" begin
    @test rf ≈ (1 - δ - 2/ϵ)/3
    @test rc ≈ 1/5
    @test maximum(pf -> _norm(pf), mpf.pfs) ≈ 1
    @test re ≈ 1/5
end

## Pos and Lie: 2 wits #2
nloc = 2
gen = CPV.Generator(nvar, nloc)

CPV.add_lf!(gen, 1)
CPV.add_lf!(gen, 2)
point = [-2, 0]
CPV.add_evidence!(gen, PosEvidence(1, 1, point, norm(point, Inf)))
point = [2, 0]
CPV.add_evidence!(gen, PosEvidence(2, 1, point, norm(point, Inf)))
point1 = [2, 0]
point2 = [4, 0]
CPV.add_evidence!(
    gen, LieDiscEvidence(
        2, 1, point1, 1, point2,
        norm(point1, Inf), norm(point2, Inf),
        norm(point2 - point1, Inf), 4.0, 1.0
    )
)

ϵ = 1e5
δ = 3.0
rf = CPV.compute_mpf_feasibility(gen, ϵ, δ, solver)[2]
mpf, rc = CPV.compute_mpf_chebyshev(gen, solver)
re = CPV.compute_mpf_evidence(gen, solver)[2]

@testset "compute mpf pos and lie: 2 wits #2" begin
    @test rf ≈ min(1 - 1/ϵ, 3 - δ)
    @test rc ≈ 1
    @test maximum(pf -> _norm(pf), mpf.pfs) ≈ 1
    @test re ≈ 3/5
end

## Pos and Lie: 2 wits #3
nloc = 2
gen = CPV.Generator(nvar, nloc)

CPV.add_lf!(gen, 1)
CPV.add_lf!(gen, 2)
point = [2, 0]
CPV.add_evidence!(gen, PosEvidence(1, 1, point, norm(point, Inf)))
point1 = [0, 2]
point2 = [4, 0]
CPV.add_evidence!(
    gen, LieDiscEvidence(
        1, 1, point1, 2, point2,
        norm(point1, Inf), norm(point2, Inf),
        norm(point2 - point1, Inf), 4.0, 1.0
    )
)
point1 = [2, 0]
point2 = [0, 0]
CPV.add_evidence!(
    gen, LieDiscEvidence(
        2, 1, point1, 1, point2,
        norm(point1, Inf), norm(point2, Inf),
        norm(point2 - point1, Inf), 4.0, 1.0
    )
)

ϵ = 1e5
δ = 0.0
rf = CPV.compute_mpf_feasibility(gen, ϵ, δ, solver)[2]
mpf, rc = CPV.compute_mpf_chebyshev(gen, solver)
re = CPV.compute_mpf_evidence(gen, solver)[2]

@testset "compute mpf pos and lie: 2 wits #3" begin
    @test rf ≈ (1 - 3*δ - 1/ϵ)/4
    @test rc ≈ 1/6
    @test maximum(pf -> _norm(pf), mpf.pfs) ≈ 1
    @test re ≈ 1/16
end

## Lie: same var
nloc = 1
gen = CPV.Generator(nvar, nloc)

CPV.add_lf!(gen, 1)
point1 = [0, 2]
point2 = [0, 2]
CPV.add_evidence!(
    gen, LieContEvidence(
        1, 1, point1, point2,
        norm(point1, Inf), norm(point2, Inf), 0.0, -Inf, 1.0, 2.0
    )
)

rc = CPV.compute_mpf_chebyshev(gen, solver)[2]
re = CPV.compute_mpf_evidence(gen, solver)[2]

@testset "compute mpf pos and lie: 2 wits #3" begin
    @test rc ≈ 2
    @test re ≈ 0
end

nothing