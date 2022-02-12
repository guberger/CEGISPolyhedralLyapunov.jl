using LinearAlgebra
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
ϵ = 1e-1
tol = eps(1.0)
δ_min = 1e-1
D = 2
M = 4

## Tests
domain = zeros(1, D)
fields = [[-101.1 99; 101 -99.1], [-101.1 -99; -101 -99.1]]
sys = CPL.LinearSystem(domain, fields)
systems = (sys,)

# flows_init = map()
# witness1 = CPL.Witness(flow1, 1)
# witness2 = CPL.Witness(flow2, 2)
# node1 = CPL.Node(witness1, 1)
# node2 = CPL.Node(witness1, 2)
# node3 = CPL.Node(witness2, 2)
# nodes_init = (node1, node2, node3)
nodes_init = CPL.Node[]

coeffs, nodes, obj_max, flag =
    CPL.process_PLF_fixed(M, D, systems, nodes_init,
                          ϵ, tol, δ_min, solver,
                          iter_max=100000,
                          output_period=100,
                          learner_output=false)

println("\nfinished-----------------------------------------------------------")