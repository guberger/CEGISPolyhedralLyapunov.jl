module ExamplePerformance

using LinearAlgebra
using Printf
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
ϵ = 10.0
nloc = 1

f = open(string(@__DIR__, "/measurements.txt"), "w")
iter = 0
max_iter = 100

for nvar in (4, 5, 6, 7, 8, 9)
    iter > max_iter && break
    ONE_ = ones(nvar, nvar)
    EYE_ = Matrix{Float64}(I, nvar, nvar)
    U = qr([float(iseven(i)) for i = 1:nvar]).Q

    a = [(k == 1 ? 1.0 : 0.0) for k = 1:nvar]
    domain1 = CPL.Cone()
    CPL.add_supp!(domain1, -a)
    A1 = U \ (ONE_ - (nvar + 1)*EYE_) * U
    domain2 = CPL.Cone()
    CPL.add_supp!(domain2, a)

    str = @sprintf("nvar = %d\n", nvar)
    print(string("---> ", str))
    print(f, str)

    for γ in (1.0, 0.1, 0.01)
        sys = CPL.System()
        CPL.add_piece_cont!(sys, domain1, 1, A1)
        A2 = U \ (ONE_ - (nvar + γ)*EYE_) * U
        CPL.add_piece_cont!(sys, domain2, 1, A2)
        for δ in (γ/50, γ)
            global iter += 1
            iter > max_iter && break
            lear = CPL.Learner(nvar, nloc, sys, τ, ϵ, δ)
            status, mpf, niter = @time CPL.learn_lyapunov!(
                lear, 1000, solver, solver, do_print=false
            )
            time = @elapsed status2, mpf, niter2 = CPL.learn_lyapunov!(
                lear, 1000, solver, solver, do_print=false
            )
            @assert niter == niter2
            @assert status == status2
            @assert status ∈ (CPL.LYAPUNOV_FOUND, CPL.LYAPUNOV_INFEASIBLE)
            σ = -maximum(real.(eigvals(A2)))
            str = @sprintf("%s | %f & %e & %.2f & %d\n",
                status, γ, σ, time, niter)
            print(string("---> ", str))
            print(f, str)
            flush(f)
        end
    end
end

close(f)

end # module