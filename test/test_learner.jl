using StaticArrays
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
DV = Val(2)

## Tests
A = [-1.0 0.0; 0.0 -1.0]
A_list = [A]

x_list = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
witnesses = map(x -> CPL.Witness(DV, x, map(A -> A*x, A_list)), x_list)
M = length(witnesses)
dict_indexes = Dict([(w, (i,)) for (i, w) in enumerate(witnesses)])
get_indexes(witness) = get(dict_indexes, witness, (0,))
G0 = Gmax = 1.0
r0 = rmin = 1.0 + 1e-5
δ, coeffs, G, r, flag = CPL.learn_PLF_params(M, witnesses, get_indexes,
                                             G0, Gmax, r0, rmin, ϵ, solver)

@testset "Learner Points: LTI infeasible" begin
    @test G == G0
    @test r == r0
    @test !flag
end

G0 = 0.25
Gmax = 1.0 + 1e-5
r0 = 4.0 - 1e-5
rmin = 0.0
δ, coeffs, G, r, flag = CPL.learn_PLF_params(M, witnesses, get_indexes,
                                             G0, Gmax, r0, rmin, ϵ, solver)

@testset "Learner Points: LTI feasible" begin
    @test G == 1.0
    @test r == r0/4
    @test flag
    @test abs(δ - 1.0) < 1e-6
end

A1 = [-1.0 0.0; 0.0 -2.0]
A2 = [-2.0 0.0; 0.0 -1.0]
A_list = [A1, A2]

x_list = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
witnesses = map(x -> CPL.Witness(DV, x, map(A -> A*x, A_list)), x_list)
M = length(witnesses)
dict_indexes = Dict([(w, (i,)) for (i, w) in enumerate(witnesses)])
get_indexes(witness) = get(dict_indexes, witness, (0,))

G0 = Gmax = 1.0
r0 = rmin = 0.0 + 1e-5
δ, coeffs, G, r, flag = CPL.learn_PLF_params(M, witnesses, get_indexes,
                                             G0, Gmax, r0, rmin, ϵ, solver)

@testset "Learner Points: SLS infeasible" begin
    @test G == G0
    @test r == r0
    @test !flag
end

G0 = 0.25
Gmax = 2.0 + 1e-5
r0 = 8.0 - 1e-5
rmin = 0.0
δ, coeffs, G, r, flag = CPL.learn_PLF_params(M, witnesses, get_indexes,
                                             G0, Gmax, r0, rmin, ϵ, solver)

@testset "Learner Points: SLS feasible" begin
    @test G == 2.0
    @test r == r0/8
    @test flag
    @test abs(δ - 1.0) < 1e-6
end

println("\nfinished-----------------------------------------------------------")