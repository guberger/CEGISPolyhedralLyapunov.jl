module ExamplePerformance_Compute

using Random
using Statistics
using LinearAlgebra
using JuMP
using Gurobi

Random.seed!(0)

include("../../src/CEGISPolyhedralLyapunov.jl")
CPL = CEGISPolyhedralLyapunov

const GUROBI_ENV = Gurobi.Env()
function solver()
    model = direct_model(
        optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV))
    )
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "Method", 2)
    return model
end
# Method: 2 = barrier (force this one, for reproducible results)

## Parameters
τ = 1/8 # 0.1
xmax = 1e4
tol_r = 1e-6
iter_max = 1000
nsample = 10
# nsample = 2

N_list = [4, 5, 6, 7, 8, 9] # dimension
γ_list = [0.5, 0.05]
M_list = [1, 2, 4] # pieces
TIMEs_list_list = Vector{Vector{Tuple{Float64,Float64}}}[]
NITERs_list_list = Vector{Vector{Tuple{Float64,Float64}}}[]
first_pass = true

for N in N_list
    println("--> N: ", N)
    local TIMEs_list = Vector{Tuple{Float64,Float64}}[]
    local NITERs_list = Vector{Tuple{Float64,Float64}}[]
    local lfs_init::Vector{Vector{Float64}} = collect(Iterators.flatten(
        ([(k == i ? s/10 : 0.0) for k in 1:N] for i in 1:N) for s in (-1, 1)
    ))
    for γ in γ_list
        println("--> γ: ", γ)
        local TIMEs = Tuple{Float64,Float64}[]
        local NITERs = Tuple{Float64,Float64}[]
        for M in M_list
            println("--> M: ", M)
            local times = Float64[]
            local niters = Int[]
            for k in 1:nsample
                local U = qr(randn(N)).Q
                local A1 = U \ (ones(N, N) - (N + 1)*I) * U
                local A2 = U \ (ones(N, N) - (N + γ)*I) * U
                local pieces =
                    CPL.Piece{Matrix{Float64},Vector{Vector{Float64}}}[]
                for q in 1:M
                    local a = randn(N)
                    normalize!(a)
                    push!(pieces, CPL.Piece(A1, [a]))
                    push!(pieces, CPL.Piece(A2, [-a]))
                end
                if first_pass
                    CPL.learn_lyapunov(
                        pieces, lfs_init, τ, 1, N, xmax, iter_max, solver,
                        tol_r=tol_r, do_print=false
                    )
                    global first_pass = false
                end
                local time = @elapsed local status, lfs = CPL.learn_lyapunov(
                    pieces, lfs_init, τ, 1, N, xmax, iter_max, solver,
                    tol_r=tol_r, do_print=false
                )
                @assert status == CPL.LYAPUNOV_FOUND
                println(status, ", ", γ, ", ", time, ", ", length(lfs))
                push!(times, time)
                push!(niters, length(lfs))
            end
            push!(TIMEs, (mean(times), std(times)))
            push!(NITERs, (mean(niters), std(niters)))
        end
        push!(TIMEs_list, TIMEs)
        push!(NITERs_list, NITERs)
    end
    push!(TIMEs_list_list, TIMEs_list)
    push!(NITERs_list_list, NITERs_list)
end

f = open(string(@__DIR__, "/measurements.txt"), "w")
println(f, "sec: N")
for N in N_list
    println(f, N)
end
println(f, "sec: γ")
for γ in γ_list
    println(f, γ)
end
println(f, "sec: M")
for M in M_list
    println(f, M)
end
println(f, "sec: TIME")
for TIMEs_list in TIMEs_list_list
    println(f, TIMEs_list)
end
println(f, "sec: NITER")
for NITERs_list in NITERs_list_list
    println(f, NITERs_list)
end
close(f)

end # module