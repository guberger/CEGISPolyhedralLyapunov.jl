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
M = 4

## Tests
domain = zeros(1, D)
fields = [[-1.0 0.0; 0.0 -1.0]]
sys = CPL.LinearSystem(domain, fields)
systems = 

flow1 = CPL.make_flows((sys,), ([-1.0, 0.0],))[1]
flow2 = CPL.make_flows((sys,), ([+1.0, 0.0],))[1]
witness1 = CPL.Witness(flow1, 1)
witness2 = CPL.Witness(flow2, 2)
node1 = CPL.Node(witness1, 1)
node2 = CPL.Node(witness1, 2)
node3 = CPL.Node(witness2, 2)
nodes = (node1, node2, node3)
δ, coeffs, flag = CPL.learn_PLF_branch(M, D, nodes, ϵ, solver)

@testset "Learner Branch: LTI infeasible" begin
    @test δ ≤ eps(100.0)
    @test !flag
end

node1 = CPL.Node(witness1, 1)
node2 = CPL.Node(witness2, 2)
nodes = (node1, node2)
δ, coeffs, flag = CPL.learn_PLF_branch(M, D, nodes, ϵ, solver)

@testset "Learner Branch: LTI feasible" begin
    @test abs(δ - 1.0) < 1e-6
    @test flag
end

domain = zeros(1, D)
fields = [[+1.0 0.0; 0.0 -2.0],
          [-2.0 0.0; 0.0 -1.0]]
sys = CPL.LinearSystem(domain, fields)
systems = (sys,)

flow = CPL.make_flows((sys,), ([-1.0, 0.0],))[1]
witness = CPL.Witness(flow, 1)
node = CPL.Node(witness, 1)
nodes = (node,)
δ, coeffs, flag = CPL.learn_PLF_branch(M, D, nodes, ϵ, solver)

@testset "Learner Branch: SLS infeasible" begin
    @test δ ≤ eps(100.0)
    @test !flag
end

domain = zeros(1, D)
fields = [[-1.0 0.0; 0.0 -2.0],
          [-2.0 0.0; 0.0 -1.0]]
sys = CPL.LinearSystem(domain, fields)
systems = (sys,)

flow1 = CPL.make_flows((sys,), ([-1.0, 0.0],))[1]
flow2 = CPL.make_flows((sys,), ([+1.0, 0.0],))[1]
witness1 = CPL.Witness(flow1, 1)
witness2 = CPL.Witness(flow2, 2)
node1 = CPL.Node(witness1, 1)
node2 = CPL.Node(witness2, 2)
nodes = (node1, node2)
δ, coeffs, flag = CPL.learn_PLF_branch(M, D, nodes, ϵ, solver)

@testset "Learner Branch: SLS feasible" begin
    @test flag
    @test abs(δ - 1.0) < 1e-6
end

println("\nfinished-----------------------------------------------------------")