module ExamplePerformance

using LinearAlgebra
using Printf
using JuMP
using Gurobi

include("../../../src/CEGISPolyhedralLyapunov.jl")
CPL = CEGISPolyhedralLyapunov
CPLA = CPL.AdaptiveComplexity
CPLP = CPL.Polyhedra

const GUROBI_ENV = Gurobi.Env()
solver = optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
)

## Parameters
ϵ = 10.0
θ = 0.1
δ = 1e-6

f = open(string(@__DIR__, "/measurements.txt"), "w")
iter = 0
max_iter = 100

# for nvar in (4, 5, 6, 7, 8, 9)
for nvar in (9,)
    iter > max_iter && break
    ONE_ = ones(nvar, nvar)
    EYE_ = Matrix{Float64}(I, nvar, nvar)
    U = qr([float(iseven(i)) for i = 1:nvar]).Q

    a = [(k == 1 ? 1.0 : 0.0) for k = 1:nvar]
    domain1 = CPLP.Cone()
    CPLP.add_supp!(domain1, -a)
    A1 = U \ (ONE_ - (nvar + 1)*EYE_) * U
    domain2 = CPLP.Cone()
    CPLP.add_supp!(domain2, a)

    str = @sprintf("nvar = %d\n", nvar)
    print(string("---> ", str))
    print(f, str)

    for γ in (1.0, 0.1, 0.01)
        for δ in (γ/50, γ)
            global iter += 1
            iter > max_iter && break
            sys = CPLA.System()
            CPLA.add_piece!(sys, domain1, A1)
            A2 = U \ (ONE_ - (nvar + γ)*EYE_) * U
            CPLA.add_piece!(sys, domain2, A2)

            lear = CPLA.Learner(nvar, sys, ϵ, θ, δ)

            sol = @time CPLA.learn_lyapunov!(
                lear, 1000, solver, do_print=false
            )
            time = @elapsed CPLA.learn_lyapunov!(
                lear, 1000, solver, do_print=false
            )
            complexity = 0
            deriv = Inf
            if sol.status == CPLA.LYAPUNOV_FOUND
                deriv = sol.val_lie_list[sol.niter]
                complexity = length(sol.vecs_list[sol.niter])
            elseif sol.status == CPLA.LYAPUNOV_INFEASIBLE
                complexity = sol.niter
            else
                error("Unexpected status")
            end
            σ = -maximum(real.(eigvals(A2)))
            str = @sprintf("%s %f | %f & %e & %.2f & %d\n",
                sol.status, deriv, γ, σ, time, complexity)
            print(string("---> ", str))
            print(f, str)
            flush(f)
        end
    end
end

close(f)

end # module