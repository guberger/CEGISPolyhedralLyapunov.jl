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
CPLP = CPL.Polyhedra
CPLV = CPL.Verifier

solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)

## Parameters
nvar = 2
domain = CPLP.Cone()
CPLP.add_supp!(domain, CPLP.Supp([1.0, 1.0]))
verif = CPLV.VerifyingProblem(nvar, domain, zeros(nvar, nvar))

vecs = [[-0.5, 0.5], [1.0, 0.0]]

x, r = CPLV.verify_pos(verif, vecs, solver)

@testset "verify pos" begin
    @test r ≈ -1/3
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
end

nvar = 2
domain = CPLP.Cone()
CPLP.add_supp!(domain, CPLP.Supp([-1.0, -1.0]))
verif = CPLV.VerifyingProblem(nvar, domain, zeros(nvar, nvar))

vecs = [[-0.5, 0.5], [1.0, 0.0]]

x, r = CPLV.verify_pos(verif, vecs, solver)

@testset "verify pos" begin
    @test r ≈ 1/3
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
end

domain = CPLP.Cone()
A = [0.0 1.0; 0.0 0.0]
verif = CPLV.VerifyingProblem(nvar, domain, A)

vecs = [[-1.0, 0.0], [1.0, 0.0]]

x, r = CPLV.verify_lie(verif, vecs, solver)

@testset "verify lie" begin
    @test r ≈ 1
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test x[2] ≈ -1.0
end

domain = CPLP.Cone()
CPLP.add_supp!(domain, CPLP.Supp([-1.0, 0.0]))
A = [1.0 0.1; 0.0 0.0]
verif = CPLV.VerifyingProblem(nvar, domain, A)

vecs = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]

x, r = CPLV.verify_lie(verif, vecs, solver)

@testset "verify lie" begin
    @test r ≈ 1.1
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test x ≈ [1, 1]
end

domain = CPLP.Cone()
CPLP.add_supp!(domain, CPLP.Supp([1.0, 0.0]))
A = [-1.0 0.1; 0.0 -1.0]
verif = CPLV.VerifyingProblem(nvar, domain, A)

x, r = CPLV.verify_lie(verif, vecs, solver)

@testset "verify lie" begin
    @test r ≈ -0.9
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
end

domain = CPLP.Cone()
A = [-101.1 99; 101 -99.1]
verif = CPLV.VerifyingProblem(nvar, domain, A)

x, r = CPLV.verify_lie(verif, vecs, solver)

@testset "verify lie" begin
    @test r ≈ 1.9
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test x ≈ [-1, -1]
end

vecs = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]

x, r = CPLV.verify_lie(verif, vecs, solver)

@testset "verify lie" begin
    @test r ≈ -0.1
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
end

nothing