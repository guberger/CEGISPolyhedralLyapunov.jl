using LinearAlgebra
using JuMP
using HiGHS
using Test
@static if isdefined(Main, :TestLocal)
    include("../../src/CEGISPolyhedralLyapunov.jl")
else
    using CEGISPolyhedralLyapunov
end
CPLP = CEGISPolyhedralLyapunov.Polyhedra
CPLA = CEGISPolyhedralLyapunov.AdaptiveComplexity

solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)

## Parameters
verif = CPLA.Verifier()
nvar = 2
domain = CPLP.Cone()
CPLP.add_supp!(domain, [1.0, 1.0])
CPLA.add_verifying_pos!(verif, nvar, domain)

vecs = [[-0.5, 0.5], [1.0, 0.0]]

x, r = CPLA.verify_pos(verif, vecs, solver)

@testset "verify pos" begin
    @test r ≈ -1/3
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
end

verif = CPLA.Verifier()
nvar = 2
domain = CPLP.Cone()
CPLP.add_supp!(domain, [-1.0, -1.0])
CPLA.add_verifying_pos!(verif, nvar, domain)

vecs = [[-0.5, 0.5], [1.0, 0.0]]

x, r = CPLA.verify_pos(verif, vecs, solver)

@testset "verify pos" begin
    @test r ≈ 1/3
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
end

verif = CPLA.Verifier()
domain = CPLP.Cone()
A = [0.0 1.0; 0.0 0.0]
CPLA.add_verifying_lie!(verif, nvar, domain, A)

vecs = [[-1.0, 0.0], [1.0, 0.0]]

x, r = CPLA.verify_lie(verif, vecs, solver)

@testset "verify lie" begin
    @test r ≈ 1
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test x[2] ≈ -1.0
end

verif = CPLA.Verifier()
domain = CPLP.Cone()
CPLP.add_supp!(domain, [-1.0, 0.0])
A = [1.0 0.1; 0.0 0.0]
CPLA.add_verifying_lie!(verif, nvar, domain, A)

vecs = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]

x, r = CPLA.verify_lie(verif, vecs, solver)

@testset "verify lie" begin
    @test r ≈ 1.1
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test x ≈ [1, 1]
end

verif = CPLA.Verifier()
domain = CPLP.Cone()
CPLP.add_supp!(domain, [1.0, 0.0])
A = [-1.0 0.1; 0.0 -1.0]
CPLA.add_verifying_lie!(verif, nvar, domain, A)

x, r = CPLA.verify_lie(verif, vecs, solver)

@testset "verify lie" begin
    @test r ≈ -0.9
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
end

verif = CPLA.Verifier()
domain = CPLP.Cone()
A = [-101.1 99; 101 -99.1]
CPLA.add_verifying_lie!(verif, nvar, domain, A)

x, r = CPLA.verify_lie(verif, vecs, solver)

@testset "verify lie" begin
    @test r ≈ 1.9
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test x ≈ [-1, -1]
end

vecs = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]

x, r = CPLA.verify_lie(verif, vecs, solver)

@testset "verify lie" begin
    @test r ≈ -0.1
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
end

nothing