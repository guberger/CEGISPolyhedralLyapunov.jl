using JuMP
using HiGHS
using Test
@static if isdefined(Main, :TestLocal)
    include("../src/CEGPolyhedralLyapunov.jl")
else
    using CEGPolyhedralLyapunov
end
CPL = CEGPolyhedralLyapunov

solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)

## Parameters
ϵ = 1e-5
D = 2
coeffs_ = Vector{VariableRef}[]
coeffs = Vector{Float64}[]

## Tests
flows = CPL.Flow[]
M = length(flows)
G0 = Gmax = 1.0
r0 = rmin = 1.0
δ, G, r, flag = CPL.learn_PLF_adaptive!(0, M, D, coeffs_, coeffs,
                                        flows, G0, Gmax, r0, rmin,
                                        ϵ, solver)

@testset "Learner Adaptive: empty flows" begin
    @test isinf(δ)
    @test G == G0
    @test r == r0
    @test flag
end

domain = zeros(1, D)
fields = [[-1.0 0.0; 0.0 -1.0]]
sys = CPL.LinearSystem(domain, fields)
systems = (sys,)

points = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
flows = map(x -> CPL.make_flow(systems, x), points)
M = length(flows)
resize!(coeffs_, M)
coeffs = [Vector{Float64}(undef, D) for i = 1:M]
G0 = Gmax = 1.0
r0 = rmin = 1.0 + 1e-5
δ, G, r, flag = CPL.learn_PLF_adaptive!(0, M, D, coeffs_, coeffs,
                                        flows, G0, Gmax, r0, rmin,
                                        ϵ, solver)

@testset "Learner Adaptive: LTI infeasible" begin
    @test G == G0
    @test r == r0
    @test !flag
end

G0 = 0.25
Gmax = 1.0 + 1e-5
r0 = 4.0 - 1e-5
rmin = 0.0
δ, G, r, flag = CPL.learn_PLF_adaptive!(0, M, D, coeffs_, coeffs,
                                        flows, G0, Gmax, r0, rmin,
                                        ϵ, solver)

@testset "Learner Adaptive: LTI feasible" begin
    @test G == 1.0
    @test r == r0/4
    @test flag
    @test abs(δ - 1.0) < 1e-6
end

domain = zeros(1, D)
fields = [[-1.0 0.0; 0.0 -2.0],
          [-2.0 0.0; 0.0 -1.0]]
sys = CPL.LinearSystem(domain, fields)
systems = (sys,)

points = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
flows = map(x -> CPL.make_flow(systems, x), points)
M = length(flows)
resize!(coeffs_, M)
coeffs = [Vector{Float64}(undef, D) for i = 1:M]

G0 = Gmax = 1.0
r0 = rmin = 0.0 + 1e-5
δ, G, r, flag = CPL.learn_PLF_adaptive!(0, M, D, coeffs_, coeffs,
                                        flows, G0, Gmax, r0, rmin,
                                        ϵ, solver)

@testset "Learner Adaptive: SLS infeasible" begin
    @test G == G0
    @test r == r0
    @test !flag
end

G0 = 0.25
Gmax = 2.0 + 1e-5
r0 = 8.0 - 1e-5
rmin = 0.0
δ, G, r, flag = CPL.learn_PLF_adaptive!(0, M, D, coeffs_, coeffs,
                                        flows, G0, Gmax, r0, rmin,
                                        ϵ, solver)

@testset "Learner Adaptive: SLS feasible" begin
    @test G == 2.0
    @test r == r0/8
    @test flag
    @test abs(δ - 1.0) < 1e-6
end

println("\nfinished-----------------------------------------------------------")