module ExamplePerformance

using LinearAlgebra
using Printf
using JuMP
using Gurobi
include("../../src/CEGPolyhedralLyapunov.jl")
CPL = CEGPolyhedralLyapunov

const GUROBI_ENV = Gurobi.Env()
solver = optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV),
    "OutputFlag"=>false)

## Parameters
G0 = 0.1
Gmax = 10.0
r0 = 0.01
rmin = 1e-6
ϵ = 1e-2
tol = -1e-5
output_period = 10

HP(D) = map(i -> i == 1 ? 1 : 0, reshape(1:D, 1, D))

f = open(string(@__DIR__, "/measurements.txt"), "w")
iter = 0
max_iter = 50

for D in (4, 5, 6, 7, 8, 9)
    iter > max_iter && break
    ONE_ = ones(D, D)
    EYE_ = Matrix{Bool}(I, D, D)
    U = qr(map(i -> (-1)^i, 1:D)).Q
    domain1 = +HP(D)
    domain2 = -HP(D)
    str = @sprintf("D = %d\n", D)
    print(string("---> ", str))
    print(f, str)
    for γ in (1, 0.1, 0.01)
        global iter += 1
        iter > max_iter && break
        fields1 = [U \ (ONE_ - (D + γ)*EYE_) * U]
        fields2 = [U \ (ONE_ - (D + 1)*EYE_) * U]
        sys1 = CPL.LinearSystem(domain1, fields1)
        sys2 = CPL.LinearSystem(domain2, fields2)
        systems = (sys1, sys2)
        flows_init = CPL.Flow[]
        coeffs, flows, deriv, flag, trace =
            @time CPL.process_PLF_adaptive(D, systems, flows_init,
                                           G0, Gmax, r0, rmin, ϵ, tol,
                                           solver, output_period=10,
                                           learner_output=false, trace=false)
        time = 
            @elapsed CPL.process_PLF_adaptive(D, systems, flows_init,
                                              G0, Gmax, r0, rmin, ϵ, tol,
                                              solver, output_period=10,
                                              learner_output=false, trace=false)
        complexity = length(coeffs)
        σ = -maximum(real.(eigvals(Matrix(sys1.fields[1]))))
        str = @sprintf("%s %f | %f & %e & %.2f & %d\n",
            flag, deriv, γ, σ, time, complexity)
        print(string("---> ", str))
        print(f, str)
    end
end

close(f)

end # module