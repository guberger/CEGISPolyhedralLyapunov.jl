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
verif = CPV.Verifier()
nvar = 2
domain = CPV.Cone()
CPV.add_supp!(domain, [1.0, 1.0])
CPV.add_verifying_pos!(verif, nvar, domain)

lins = [[-0.5, 0.5], [1.0, 0.0]]
lfs = [CPV.LinForm(lin) for lin in lins]

x, r = CPV.verify_pos(verif, lfs, solver)

@testset "verify pos" begin
    @test r ≈ -1/3
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
end

verif = CPV.Verifier()
nvar = 2
domain = CPV.Cone()
CPV.add_supp!(domain, [-1.0, -1.0])
CPV.add_verifying_pos!(verif, nvar, domain)

lins = [[-0.5, 0.5], [1.0, 0.0]]
lfs = [CPV.LinForm(lin) for lin in lins]

x, r = CPV.verify_pos(verif, lfs, solver)

@testset "verify pos" begin
    @test r ≈ 1/3
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
end

verif = CPV.Verifier()
domain = CPV.Cone()
A = [0.0 1.0; 0.0 0.0]
CPV.add_verifying_lie!(verif, nvar, domain, A)

lins = [[-1.0, 0.0], [1.0, 0.0]]
lfs = [CPV.LinForm(lin) for lin in lins]

x, r = CPV.verify_lie(verif, lfs, solver)

@testset "verify lie" begin
    @test r ≈ 1
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test x[2] ≈ -1.0
end

verif = CPV.Verifier()
domain = CPV.Cone()
CPV.add_supp!(domain, [-1.0, 0.0])
A = [1.0 0.1; 0.0 0.0]
CPV.add_verifying_lie!(verif, nvar, domain, A)

lins = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
lfs = [CPV.LinForm(lin) for lin in lins]

x, r = CPV.verify_lie(verif, lfs, solver)

@testset "verify lie" begin
    @test r ≈ 1.1
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test x ≈ [1, 1]
end

verif = CPV.Verifier()
domain = CPV.Cone()
CPV.add_supp!(domain, [1.0, 0.0])
A = [-1.0 0.1; 0.0 -1.0]
CPV.add_verifying_lie!(verif, nvar, domain, A)

x, r = CPV.verify_lie(verif, lfs, solver)

@testset "verify lie" begin
    @test r ≈ -0.9
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
end

verif = CPV.Verifier()
domain = CPV.Cone()
A = [-101.1 99; 101 -99.1]
CPV.add_verifying_lie!(verif, nvar, domain, A)

x, r = CPV.verify_lie(verif, lfs, solver)

@testset "verify lie" begin
    @test r ≈ 1.9
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test x ≈ [-1, -1]
end

lins = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]
lfs = [CPV.LinForm(lin) for lin in lins]

x, r = CPV.verify_lie(verif, lfs, solver)

@testset "verify lie" begin
    @test r ≈ -0.1
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
end

nothing