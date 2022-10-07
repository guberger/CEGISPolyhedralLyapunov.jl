module ExamplePerformance

using LinearAlgebra
using JuMP
using Gurobi

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
tol_r = 1e-5
iter_max = 1000

imeas = 0
max_meas = 5

N_list = [4, 5, 6, 7, 8, 9]
Ts_list = Vector{Float64}[]
NITERs_list = Vector{Int}[]

piece1::Bool = true

for N in N_list
    imeas > max_meas && break
    local Ts = Float64[]
    local NITERs = Int[]
    local U = qr([float(iseven(i)) for i in 1:N]).Q
    piece1 = CPL.Piece(
        U \ (ones(N, N) - (N + 1)*I) * U,
        [[(k == 1 ? -1.0 : 0.0) for k in 1:N]]
    )
    local lfs_init = collect(Iterators.flatten(
        ([(k == i ? s/10 : 0.0) for k in 1:N] for i in 1:N) for s in (-1, 1)
    ))
    println("--> N: ", N)

    for γ in (1.0, 0.1, 0.01)
        local piece2 = CPL.Piece(
            U \ (ones(N, N) - (N + γ)*I) * U,
            [[(k == 1 ? 1.0 : 0.0) for k in 1:N]]
        )
        local pieces = [piece1, piece2]
        global imeas += 1
        imeas > max_meas && break
        local status1, lfs1 = CPL.learn_lyapunov(
            pieces, lfs_init, τ, N, xmax, iter_max, solver,
            tol_r=tol_r, do_print=false
        )
        local time = @elapsed local status2, lfs2 = CPL.learn_lyapunov(
            pieces, lfs_init, τ, N, xmax, iter_max, solver,
            tol_r=tol_r, do_print=false
        )
        @assert status1 == status2
        @assert length(lfs1) == length(lfs2)
        @assert status1 ∈ (CPL.LYAPUNOV_FOUND, CPL.LYAPUNOV_INFEASIBLE)
        println(status2, ", ", γ, ", ", time, ", ", length(lfs2))
        push!(Ts, time)
        push!(NITERs, length(lfs2))
    end
    push!(Ts_list, Ts)
    push!(NITERs_list, NITERs)
end

f = open(string(@__DIR__, "/measurements.txt"), "w")
println(Ts_list)
println(NITERs_list)
close(f)

end # module