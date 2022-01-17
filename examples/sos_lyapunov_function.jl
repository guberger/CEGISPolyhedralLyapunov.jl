using LinearAlgebra

μ = 0.05
V = [1 2; 3 4]
H1 = [-1.0 1.0;
      0.0 -μ]
H2 = [-1.0 -1.0;
      0.0 -μ]
H3 = [-μ 0.0;
      1.0 -1.0]
H_list = [A1, A2, A3]
A_list = map(H -> V \ H * V, H_list)
D = 2

using DynamicPolynomials
using SumOfSquares
using MosekTools

solver = MosekTools.Optimizer
model = SOSModel(solver)

@polyvar x[1:D]
dh = 2
X = monomials(x, 2*dh)

@variable(model, V, Poly(X))
@constraint(model, V ≥ sum(x.^(2*dh)))

for A in A_list
    dV = dot(differentiate(V, x), A*x)
    @constraint(model, dV ≤ 0)
end

optimize!(model)

display(solution_summary(model))

using JuMP
using Gurobi
include("../src/CEGARLearningCLF.jl")
CLC = CEGARLearningCLF

meth_learn = CLC.LearnPolyhedralPoints{D}()
meth_verify = CLC.VerifyPolyhedralMultiple{D}()
prob = CLC.CEGARProblem{D}(A_list, meth_learn, meth_verify)
G0 = 0.1
Gmax = 10.0
r0 = 0.01
rmin = 1e-6
params = (tol_faces=1e-5, tol_deriv=1e-5,
          print_period_1=1, print_period_2=1,
          do_trace=true)
solver = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag"=>false)

x_list = [[1.0,0.0], [0.0,1.0]]

## Solving
c_list, x_dx_list, deriv, flag, trace = CLC.process_lyapunov_function(
    prob, x_list, G0, Gmax, r0, rmin, params, solver)

using PyPlot

fig = figure(0, figsize=(12, 10))
ax = fig.add_subplot(aspect="equal")
xlims = (-2, 2)
ylims = (-2, 2)
ax.set_xlim(xlims...)
ax.set_ylim(ylims...)

np = 600
xf1 = range(xlims[1], stop = xlims[2], length = np)
xf2 = range(ylims[1], stop = ylims[2], length = np)
Xftemp = collect(Iterators.product(xf1, xf2))
Xf = map(x -> [x...], Xftemp)
X1 = map(x -> x[1], Xf)
X2 = map(x -> x[2], Xf)

norm_poly(x) = maximum(c -> abs(c'*x), c_list)
Z = map(x -> norm_poly(x), Xf)
zm = min(minimum(Z[1, :]), minimum(Z[:, 1]))/1.2

ax.contour(X1, X2, Z, levels=(-1, zm), colors="gold")
h = ax.contourf(X1, X2, Z, levels=(-1, zm), colors = "yellow")

N = 20
angles = range(0.0, stop=2π, length=N + 1)[1:N]
x_list = map(θ -> [cos(θ), sin(θ)], angles)

α = 0.2

for x in x_list
    nx = norm_poly(x)
    xs = x*(zm/nx)
    ax.plot(xs..., marker=".", ms=15, c="red")
    for A in A_list
        dx = A*x
        dxs = dx/norm(x)
        ys = xs + α*dxs
        ax.plot((xs[1], ys[1]), (xs[2], ys[2]), c="green", lw=2)
    end
end
