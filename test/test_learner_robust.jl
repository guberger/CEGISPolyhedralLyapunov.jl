using JuMP
using HiGHS
using Test
@static if isdefined(Main, :TestLocal)
    include("../src/CEGPolyhedralLyapunov.jl")
else
    using CEGPolyhedralLyapunov
end
CPL = CEGPolyhedralLyapunov

# Temporary fix
function HiGHS._check_ret(ret::Cint) 
    if ret != Cint(0) && ret != Cint(1)
        error(
            "Encountered an error in HiGHS (Status $(ret)). Check the log " * 
            "for details.", 
        ) 
    end 
    return 
end 

solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)

## Parameters
ϵ = 1e-5
D = 2

## Tests
domain = zeros(1, D)
fields = [[-1.0 0.0; 0.0 -1.0]]
sys = CPL.LinearSystem(domain, fields)
systems = (sys,)

points = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
flows = CPL.make_flows(systems, points)
G0 = Gmax = 1.0
r0 = rmin = 1.0 + 1e-5
δ, coeffs, G, r, flag = CPL.learn_PLF_robust(D, flows,
                                             G0, Gmax, r0, rmin, ϵ, solver)

@testset "Learner Robust: LTI infeasible" begin
    @test G == G0
    @test r == r0
    @test !flag
end

G0 = 0.25
Gmax = 1.0 + 1e-5
r0 = 4.0 - 1e-5
rmin = 0.0
δ, coeffs, G, r, flag = CPL.learn_PLF_robust(D, flows,
                                             G0, Gmax, r0, rmin, ϵ, solver)

@testset "Learner Robust: LTI feasible" begin
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
flows = CPL.make_flows(systems, points)

G0 = Gmax = 1.0
r0 = rmin = 0.0 + 1e-5
δ, coeffs, G, r, flag = CPL.learn_PLF_robust(D, flows,
                                             G0, Gmax, r0, rmin, ϵ, solver)

@testset "Learner Robust: SLS infeasible" begin
    @test G == G0
    @test r == r0
    @test !flag
end

G0 = 0.25
Gmax = 2.0 + 1e-5
r0 = 8.0 - 1e-5
rmin = 0.0
δ, coeffs, G, r, flag = CPL.learn_PLF_robust(D, flows,
                                             G0, Gmax, r0, rmin, ϵ, solver)

@testset "Learner Robust: SLS feasible" begin
    @test G == 2.0
    @test r == r0/8
    @test flag
    @test abs(δ - 1.0) < 1e-6
end

println("\nfinished-----------------------------------------------------------")