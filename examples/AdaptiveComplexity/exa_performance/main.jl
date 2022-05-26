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

HP(nvar) = map(i -> i == 1 ? 1 : 0, reshape(1:nvar, 1, nvar))

f = open(string(@__DIR__, "/measurements.txt"), "w")
iter = 0
max_iter = 5

for nvar in (4, 5, 6, 7, 8, 9)
    iter > max_iter && break
    ONE_ = ones(nvar, nvar)
    EYE_ = Matrix{Float64}(I, nvar, nvar)
    U = qr([float(iseven(i)) for i = 1:nvar]).Q

    prob = CPLA.LearningProblem(nvar, ϵ, θ, δ)
    CPLA.add_G!(prob, 1/θ)

    a = [(k == 1 ? 1.0 : 0.0) for k = 1:nvar]
    domain = CPLP.Cone()
    CPLP.add_supp!(domain, CPLP.Supp(-a))
    A = U \ (ONE_ - (nvar + 1)*EYE_) * U
    CPLA.add_system!(prob, domain, A)
    domain = CPLP.Cone()
    CPLP.add_supp!(domain, CPLP.Supp(a))

    str = @sprintf("nvar = %d\n", nvar)
    print(string("---> ", str))
    print(f, str)

    for γ in (1, 0.1, 0.01)
        global iter += 1
        iter > max_iter && break
        A = U \ (ONE_ - (nvar + γ)*EYE_) * U
        CPLA.add_system!(prob, domain, A)

        sol = @time CPLA.learn_lyapunov!(prob, 1000, solver)
        time = @elapsed CPLA.learn_lyapunov!(prob, 1000, solver)
        flag = sol.status == CPLA.LYAPUNOV_FOUND
        deriv = sol.val_lie_list[sol.niter]
        
        complexity = length(sol.vecs_list[sol.niter])
        σ = -maximum(real.(eigvals(A)))
        str = @sprintf("%s %f | %f & %e & %.2f & %d\n",
            flag, deriv, γ, σ, time, complexity)
        print(string("---> ", str))
        print(f, str)
        flush(f)
    end
end

close(f)

end # module