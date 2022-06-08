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
MultiPolyFunc = CPV.MultiPolyFunc
PosPredicate = CPV.PosPredicate
LieDiscPredicate = CPV.LieDiscPredicate
LieContPredicate = CPV.LieContPredicate

solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)

## Parameters
nvar = 2

## Pos false #1 #2
verif = CPV.Verifier()
domain = Cone()
CPV.add_supp!(domain, [1.0, 1.0])
CPV.add_predicate!(verif, PosPredicate(nvar, domain, 1))

mpf = MultiPolyFunc(2)
lins = [[-0.5, 0.5], [1.0, 0.0]]
for lin in lins
    CPV.add_lf!(mpf, 1, lin)
end

x, r, loc = CPV.verify_pos(verif, mpf, solver)

@testset "verify pos false #1" begin
    @test r ≈ 1/3
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test loc == 1
end

domain = Cone()
lin = [4.0, 0.0]
CPV.add_lf!(mpf, 2, lin)
CPV.add_predicate!(verif, PosPredicate(nvar, domain, 2))

x, r, loc = CPV.verify_pos(verif, mpf, solver)

@testset "verify pos false #2" begin
    @test r ≈ 2
    @test loc == 2
end

# Pos true
verif = CPV.Verifier()
domain = Cone()
CPV.add_supp!(domain, [-1.0, -1.0])
CPV.add_predicate!(verif, PosPredicate(nvar, domain, 1))

mpf = MultiPolyFunc(2)
lins = [[-0.5, 0.5], [1.0, 0.0]]
for lin in lins
    CPV.add_lf!(mpf, 1, lin)
end

x, r, loc = CPV.verify_pos(verif, mpf, solver)

@testset "verify pos true" begin
    @test r ≈ -1/3
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test loc == 1
end

## Lie disc false #1

verif = CPV.Verifier()
domain = Cone()
A = [1.0 1.0; 0.0 1.0]
CPV.add_predicate!(verif, LieDiscPredicate(nvar, domain, 1, A, 1))

mpf = MultiPolyFunc(2)
lins = [[-1.0, 0.0], [1.0, 0.0]]
for lin in lins
    CPV.add_lf!(mpf, 1, lin)
end

x, r, loc = CPV.verify_lie_disc(verif, mpf, solver)

@testset "verify lie disc false #1" begin
    @test r ≈ 1
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test prod(x) ≈ 1
    @test loc == 1
end

## Lie disc false #2
verif = CPV.Verifier()
domain = Cone()
CPV.add_supp!(domain, [-1.0, 0.0])
A = [2.0 0.1; 0.0 1.0]
CPV.add_predicate!(verif, LieDiscPredicate(nvar, domain, 1, A, 1))

mpf = MultiPolyFunc(2)
lins = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
for lin in lins
    CPV.add_lf!(mpf, 1, lin)
end

x, r, loc = CPV.verify_lie_disc(verif, mpf, solver)

@testset "verify lie disc false #2" begin
    @test r ≈ 1.1
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test x ≈ [1, 1]
    @test loc == 1
end

## Lie disc true #1
verif = CPV.Verifier()
domain = Cone()
CPV.add_supp!(domain, [1.0, 0.0])
A = [0.0 0.1; 0.0 0.0]
CPV.add_predicate!(verif, LieDiscPredicate(nvar, domain, 1, A, 1))

mpf = MultiPolyFunc(2)
lins = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
for lin in lins
    CPV.add_lf!(mpf, 1, lin)
end

x, r, loc = CPV.verify_lie_disc(verif, mpf, solver)

@testset "verify lie disc true #1" begin
    @test r ≈ -0.9
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test loc == 1
end

## Lie disc true #2 #3
verif = CPV.Verifier()
domain = Cone()
CPV.add_supp!(domain, [1.0, 0.0])
A = [0.0 0.5; 0.5 0.0]
CPV.add_predicate!(verif, LieDiscPredicate(nvar, domain, 1, A, 1))

mpf = MultiPolyFunc(2)
lins = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
for lin in lins
    CPV.add_lf!(mpf, 1, lin)
end

x, r, loc = CPV.verify_lie_disc(verif, mpf, solver)

@testset "verify lie disc true #2" begin
    @test r ≈ -0.5
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test x ≈ [-1, -1]
    @test loc == 1
end

mpf = MultiPolyFunc(2)
lins = [[-1.0, -1.0], [-1.0, 1.0]]
for lin in lins
    CPV.add_lf!(mpf, 1, lin)
end

x, r, loc = CPV.verify_lie_disc(verif, mpf, solver)

@testset "verify lie disc true #3" begin
    @test r ≈ -0.5
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test loc == 1
end

## Lie disc multiple #1
verif = CPV.Verifier()
domain = Cone()
CPV.add_supp!(domain, [-1.0, -1.0])
CPV.add_supp!(domain, [-1.0, 1.0])
A = [0.5 0.25; 0.0 1.0]
CPV.add_predicate!(verif, LieDiscPredicate(nvar, domain, 2, A, 1))

mpf = MultiPolyFunc(2)
lins = [[1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
for lin in lins
    CPV.add_lf!(mpf, 1, lin)
    CPV.add_lf!(mpf, 2, lin)
end

x, r, loc = CPV.verify_lie_disc(verif, mpf, solver)

@testset "verify lie disc multiple #1" begin
    @test r ≈ 0
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test loc == 2
end

mpf = MultiPolyFunc(2)
lins = [[1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
for lin in lins
    CPV.add_lf!(mpf, 2, lin)
end
CPV.add_lf!(mpf, 1, lins[1])

x, r, loc = CPV.verify_lie_disc(verif, mpf, solver)

@testset "verify lie disc multiple #2" begin
    @test r ≈ -0.25
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test loc == 2
end

## Lie cont false #1
verif = CPV.Verifier()
domain = Cone()
A = [0.0 1.0; 0.0 0.0]
CPV.add_predicate!(verif, LieContPredicate(nvar, domain, 1, A))

mpf = MultiPolyFunc(2)
lins = [[-1.0, 0.0], [1.0, 0.0]]
for lin in lins
    CPV.add_lf!(mpf, 1, lin)
end

x, r, loc = CPV.verify_lie_cont(verif, mpf, solver)

@testset "verify lie cont false #1" begin
    @test r ≈ 1
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test prod(x) ≈ 1
    @test loc == 1
end

## Lie cont false #2
verif = CPV.Verifier()
domain = Cone()
CPV.add_supp!(domain, [-1.0, 0.0])
A = [1.0 0.1; 0.0 0.0]
CPV.add_predicate!(verif, LieContPredicate(nvar, domain, 1, A))

mpf = MultiPolyFunc(2)
lins = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
for lin in lins
    CPV.add_lf!(mpf, 1, lin)
end

x, r, loc = CPV.verify_lie_cont(verif, mpf, solver)

@testset "verify lie cont false #2" begin
    @test r ≈ 1.1
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test x ≈ [1, 1]
    @test loc == 1
end

## Lie cont true #1
verif = CPV.Verifier()
domain = Cone()
CPV.add_supp!(domain, [1.0, 0.0])
A = [-1.0 0.1; 0.0 -1.0]
CPV.add_predicate!(verif, LieContPredicate(nvar, domain, 1, A))

mpf = MultiPolyFunc(2)
lins = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
for lin in lins
    CPV.add_lf!(mpf, 1, lin)
end

x, r, loc = CPV.verify_lie_cont(verif, mpf, solver)

@testset "verify lie cont true #1" begin
    @test r ≈ -0.9
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test loc == 1
end

## Lie cont false #3 true #3
verif = CPV.Verifier()
domain = Cone()
CPV.add_supp!(domain, [1.0, 0.0])
A = [-101.1 99; 101 -99.1]
CPV.add_predicate!(verif, LieContPredicate(nvar, domain, 1, A))

mpf = MultiPolyFunc(2)
lins = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
for lin in lins
    CPV.add_lf!(mpf, 1, lin)
end

x, r, loc = CPV.verify_lie_cont(verif, mpf, solver)

@testset "verify lie cont false #3" begin
    @test r ≈ 1.9
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test x ≈ [-1, -1]
    @test loc == 1
end

mpf = MultiPolyFunc(2)
lins = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]
for lin in lins
    CPV.add_lf!(mpf, 1, lin)
end

x, r, loc = CPV.verify_lie_cont(verif, mpf, solver)

@testset "verify lie cont true #3" begin
    @test r ≈ -0.1
    @test norm(x, Inf) ≈ 1
    @test x ∈ domain
    @test loc == 1
end

nothing