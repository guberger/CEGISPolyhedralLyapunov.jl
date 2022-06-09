module ExamplePerformance

using LinearAlgebra
using Printf
using JuMP
using Gurobi

include("../../src/CEGISPolyhedralVerification.jl")
CPV = CEGISPolyhedralVerification

# const GUROBI_ENV = Gurobi.Env()
# solver = optimizer_with_attributes(
#     () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
# )

## Parameters
τ = 1/8 # 0.1
ϵ = 10.0
nloc = 1

nvar = 7
γ = 0.01
δ = γ/50

rec = Tuple{Int,CPV.StatusCode}[]
tracerec_rec = CPV.TraceRecorder[]

ONE_ = ones(nvar, nvar)
EYE_ = Matrix{Float64}(I, nvar, nvar)
U = qr([float(iseven(i)) for i = 1:nvar]).Q

sys = CPV.System()
a = [(k == 1 ? 1.0 : 0.0) for k = 1:nvar]
domain1 = CPV.Cone()
CPV.add_supp!(domain1, -a)
A1 = U \ (ONE_ - (nvar + 1)*EYE_) * U
CPV.add_piece_cont!(sys, domain1, 1, A1)
domain2 = CPV.Cone()
CPV.add_supp!(domain2, a)
A2 = U \ (ONE_ - (nvar + γ)*EYE_) * U
CPV.add_piece_cont!(sys, domain2, 1, A2)

display(sys)

for i = 1:100
    display(i)
    GUROBI_ENV = Gurobi.Env()
    solver = optimizer_with_attributes(
        () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
    )

    tracerec = CPV.TraceRecorder()
    push!(tracerec_rec, tracerec)

    lear = CPV.Learner(nvar, nloc, sys, τ, ϵ, δ)
    status, mpf, niter = CPV.learn_lyapunov!(
        lear, 1000, solver, do_print=true, tracerec=tracerec
    )

    push!(rec, (niter, status))
    display(rec)

    if i > 1 && rec[i - 1][1] != niter
        break
    end
end

for lf in tracerec_rec[end - 1].mpf_list[91].pfs[1].lfs
    print(lf.lin)
end
println("\n")
for lf in tracerec_rec[end].mpf_list[91].pfs[1].lfs
    print(lf.lin)
end

print(tracerec_rec[end - 1].liecont_evids_list[91])
println("\n")
print(tracerec_rec[end].liecont_evids_list[91])

end # module