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
M = 4
coeffs_cube = ϵ.*CPL.hypercube(D)
meth = CPL.Chebyshev()

## Tests
nodes = CPL.Node[]
M0 = length(coeffs_cube)
M1 = M + M0
coeffs = [Vector{Float64}(undef, D) for i = 1:M1]
for i = 1:M0
    copyto!(coeffs[i], coeffs_cube[i])
end
δ, flag = CPL.learn_PLF_fixed!(meth, M0, M, D, coeffs,
                               nodes, solver, output_pad=4)

@testset "Learner Fixed Chebyshev: empty flows" begin
    @test isinf(δ)
    @test flag
end

domain = zeros(1, D)
fields = [[-1.0 0.0; 0.0 -1.0]]
sys = CPL.LinearSystem(domain, fields)

flow1 = CPL.make_flow((sys,), [-1, 0])
flow2 = CPL.make_flow((sys,), [+1, 0])
nodes = CPL.Node[]
for flow in (flow1, flow2)
    for k = 1:M0
        local witness = CPL.Witness(flow, k)
        for i = M0+1:M1
            local node = CPL.Node(witness, i)
            push!(nodes, node)
        end
    end
end
δ, flag = CPL.learn_PLF_fixed!(meth, M0, M, D, coeffs,
                               nodes, solver, output_pad=4)

@testset "Learner Fixed Chebyshev: LTI infeasible" begin
    @test δ < 0
    @test flag
end

witness1 = CPL.Witness(flow1, M0 + 1)
witness2 = CPL.Witness(flow2, M0 + 2)
node1 = CPL.Node(witness1, M0 + 1)
node2 = CPL.Node(witness2, M0 + 2)
nodes = (node1, node2)
δ, flag = CPL.learn_PLF_fixed!(meth, M0, M, D, coeffs,
                               nodes, solver, output_pad=4)

@testset "Learner Fixed Chebyshev: LTI feasible" begin
    @test abs(δ - 0.5) < 10*eps(0.5)
    @test flag
end

domain = zeros(1, D)
fields = [[+1.0 0.0; 0.0 -2.0],
          [-2.0 0.0; 0.0 -1.0]]
sys = CPL.LinearSystem(domain, fields)
systems = (sys,)

flow = CPL.make_flow((sys,), [-1, 0])
witness = CPL.Witness(flow, M0 + 1)
nodes = [CPL.Node(witness, M0 + 1)]
for k = 1:M0
    local witness = CPL.Witness(flow, k)
    local node = CPL.Node(witness, M0 + 1)
    push!(nodes, node)
end
δ, flag = CPL.learn_PLF_fixed!(meth, M0, M, D, coeffs,
                               nodes, solver, output_pad=4)

@testset "Learner Fixed Chebyshev: SLS infeasible" begin
    @test abs(δ + ϵ/2) < 10*eps(ϵ/2)
    @test flag
end

domain = zeros(1, D)
fields = [[-1.0 0.0; 0.0 -2.0],
          [-2.0 0.0; 0.0 -1.0]]
sys = CPL.LinearSystem(domain, fields)

flow1 = CPL.make_flow((sys,), [-1, 0])
flow2 = CPL.make_flow((sys,), [+1, 0])
witness1 = CPL.Witness(flow1, M0 + 1)
witness2 = CPL.Witness(flow2, M0 + 2)
node1 = CPL.Node(witness1, M0 + 1)
node2 = CPL.Node(witness2, M0 + 2)
nodes = (node1, node2)
δ, flag = CPL.learn_PLF_fixed!(meth, M0, M, D, coeffs,
                               nodes, solver, output_pad=4)

@testset "Learner Fixed Chebyshev: SLS feasible" begin
    @test flag
    @test abs(δ - 0.5) < 10*eps(0.5)
end

println("\nfinished-----------------------------------------------------------")