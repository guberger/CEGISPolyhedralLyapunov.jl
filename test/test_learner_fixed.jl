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
# function HiGHS._check_ret(ret::Cint) 
#     if ret != Cint(0) && ret != Cint(1)
#         error(
#             "Encountered an error in HiGHS (Status $(ret)). Check the log " * 
#             "for details.", 
#         ) 
#     end 
#     return 
# end

solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)

## Parameters
ϵ = 1e-5
D = 2
M = 4
coeffs_cube = ϵ.*CPL.hypercube(D)

## Tests
coeffs = [zeros(D) for i = 1:M]
append!(coeffs, coeffs_cube)
nodes = CPL.Node[]
coeffs_tmp = copy(coeffs)
δ, flag = CPL.learn_PLF_fixed!(M, D, coeffs, nodes, solver)

@testset "Learner Fixed: empty flows" begin
    @test isinf(δ)
    @test flag
    @test all(coeffs .== coeffs_tmp)
end

domain = zeros(1, D)
fields = [[-1.0 0.0; 0.0 -1.0]]
sys = CPL.LinearSystem(domain, fields)

flow1 = CPL.make_flow((sys,), [-1, 0])
flow2 = CPL.make_flow((sys,), [+1, 0])
nodes = CPL.Node[]
for flow in (flow1, flow2)
    for k = M+1:2*M
        local witness = CPL.Witness(flow, k)
        for i = 1:M
            local node = CPL.Node(witness, i)
            push!(nodes, node)
        end
    end
end
δ, flag = CPL.learn_PLF_fixed!(M, D, coeffs, nodes, solver)

@testset "Learner Fixed: LTI infeasible" begin
    @test δ < 0
    @test flag
end

witness1 = CPL.Witness(flow1, 1)
witness2 = CPL.Witness(flow2, 2)
node1 = CPL.Node(witness1, 1)
node2 = CPL.Node(witness2, 2)
nodes = (node1, node2)
δ, flag = CPL.learn_PLF_fixed!(M, D, coeffs, nodes, solver)

@testset "Learner Fixed: LTI feasible" begin
    @test abs(δ - 1.0) < 1e-6
    @test flag
end

domain = zeros(1, D)
fields = [[+1.0 0.0; 0.0 -2.0],
          [-2.0 0.0; 0.0 -1.0]]
sys = CPL.LinearSystem(domain, fields)
systems = (sys,)

flow = CPL.make_flow((sys,), [-1, 0])
witness = CPL.Witness(flow, 1)
nodes = [CPL.Node(witness, 1)]
for k = M+1:2*M
    local witness = CPL.Witness(flow, k)
    local node = CPL.Node(witness, 1)
    push!(nodes, node)
end
δ, flag = CPL.learn_PLF_fixed!(M, D, coeffs, nodes, solver)

@testset "Learner Fixed: SLS infeasible" begin
    @test δ < 0
    @test flag
end

domain = zeros(1, D)
fields = [[-1.0 0.0; 0.0 -2.0],
          [-2.0 0.0; 0.0 -1.0]]
sys = CPL.LinearSystem(domain, fields)

flow1 = CPL.make_flow((sys,), [-1, 0])
flow2 = CPL.make_flow((sys,), [+1, 0])
witness1 = CPL.Witness(flow1, 1)
witness2 = CPL.Witness(flow2, 2)
node1 = CPL.Node(witness1, 1)
node2 = CPL.Node(witness2, 2)
nodes = (node1, node2)
δ, flag = CPL.learn_PLF_fixed!(M, D, coeffs, nodes, solver)

@testset "Learner Fixed: SLS feasible" begin
    @test flag
    @test abs(δ - 1.0) < 1e-6
end

println("\nfinished-----------------------------------------------------------")