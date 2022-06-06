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
Cone = CPV.Cone
LinForm = CPV.LinForm
PolyFunc = CPV.PolyFunc

solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)

## Parameters
nvar = 2
_EYE_ = Matrix{Bool}(I, nvar, nvar)

verif = CPV.Verifier()
domain = Cone()
CPV.add_supp!(domain, [1.0, 1.0])
CPV.add_predicate_pos!(verif, nvar, domain, 1)

lins = [[-0.5, 0.5], [1.0, 0.0]]
lfs = [LinForm(lin) for lin in lins]
polyf = PolyFunc(lfs, [BitSet(eachindex(lfs))])

x, r = CPV.verify_pos(verif, polyf, solver)

@testset "verify pos" begin
    @test r ≈ 1/3
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
end

verif = CPV.Verifier()
domain = Cone()
CPV.add_supp!(domain, [-1.0, -1.0])
CPV.add_predicate_pos!(verif, nvar, domain, 1)

lins = [[-0.5, 0.5], [1.0, 0.0]]
lfs = [LinForm(lin) for lin in lins]
polyf = PolyFunc(lfs, [BitSet(eachindex(lfs))])

x, r = CPV.verify_pos(verif, polyf, solver)

@testset "verify pos" begin
    @test r ≈ -1/3
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
end

verif = CPV.Verifier()
domain = Cone()
DA = [0.0 1.0; 0.0 0.0]
τ = 1e-2
A = _EYE_ + τ*DA
CPV.add_predicate_lie!(verif, nvar, domain, 1, A, 1)

lins = [[-1.0, 0.0], [1.0, 0.0]]
lfs = [LinForm(lin) for lin in lins]
polyf = PolyFunc(lfs, [BitSet(eachindex(lfs))])

x, r = CPV.verify_lie(verif, polyf, solver)

@testset "verify lie" begin
    @test r ≈ τ
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test prod(x) ≈ 1
end

verif = CPV.Verifier()
domain = Cone()
CPV.add_supp!(domain, [-1.0, 0.0])
DA = [1.0 0.1; 0.0 0.0]
τ = 1e-2
A = _EYE_ + τ*DA
CPV.add_predicate_lie!(verif, nvar, domain, 1, A, 1)

lins = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
lfs = [LinForm(lin) for lin in lins]
polyf = PolyFunc(lfs, [BitSet(eachindex(lfs))])

x, r = CPV.verify_lie(verif, polyf, solver)

@testset "verify lie" begin
    @test r ≈ 1.1*τ
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test x ≈ [1, 1]
end

verif = CPV.Verifier()
domain = Cone()
CPV.add_supp!(domain, [1.0, 0.0])
DA = [-1.0 0.1; 0.0 -1.0]
τ = 1e-2
A = _EYE_ + τ*DA
CPV.add_predicate_lie!(verif, nvar, domain, 1, A, 1)

x, r = CPV.verify_lie(verif, polyf, solver)

@testset "verify lie" begin
    @test r ≈ -0.9*τ
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
end

verif = CPV.Verifier()
domain = Cone()
CPV.add_supp!(domain, [1.0, 0.0])
DA = [-101.1 99; 101 -99.1]
τ = 1e-4
A = _EYE_ + τ*DA
CPV.add_predicate_lie!(verif, nvar, domain, 1, A, 1)

x, r = CPV.verify_lie(verif, polyf, solver)

@testset "verify lie" begin
    @test r ≈ 1.9*τ
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test x ≈ [-1, -1]
end

lins = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]
lfs = [LinForm(lin) for lin in lins]
polyf = PolyFunc(lfs, [BitSet(eachindex(lfs))])

x, r = CPV.verify_lie(verif, polyf, solver)

@testset "verify lie" begin
    @test r ≈ -0.1*τ
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
end

verif = CPV.Verifier()
domain = Cone()
CPV.add_supp!(domain, [-1.0, -1.0])
CPV.add_supp!(domain, [-1.0, 1.0])
A = [0.5 0.25; 0.0 1.0]
CPV.add_predicate_lie!(verif, nvar, domain, 1, A, 2)

lins = [[1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
lfs = [LinForm(lin) for lin in lins]
polyf = PolyFunc(lfs, [BitSet(eachindex(lfs)), BitSet(eachindex(lfs))])

x, r = CPV.verify_lie(verif, polyf, solver)

@testset "verify lie" begin
    @test r ≈ 0
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
end

lins = [[1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
lfs = [LinForm(lin) for lin in lins]
polyf = PolyFunc(lfs, [BitSet(eachindex(lfs)), BitSet(1)])

x, r = CPV.verify_lie(verif, polyf, solver)

@testset "verify lie" begin
    @test r ≈ -0.25
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
end

nothing