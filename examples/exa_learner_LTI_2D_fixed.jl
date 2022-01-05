module TestMain

using Random
using JuMP
using Gurobi
using PyPlot
include("../src/CEGARLearningCLF.jl")
CLC = CEGARLearningCLF

Random.seed!(0)

## Parameters
n_piece = 5
method = CLC.LearnPolyhedralFixed{2}(n_piece)
A = [-0.1 1.0; -1.0 -0.1]
A = [-0.2 2.0; -0.5 -0.2]
A = [-0.3 0.0; -0.5 -0.3]
# A = [-1.0 0.0; 0.0 -1.0]
A_list = [A]
G0 = 0.1
Gmax = 200
r0 = 0.01
rmin = 1e-6
tol_faces = 1e-5
solver = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag"=>true)

N = 40
x_list = [randn(2) for i = 1:N]
# x_list = [[1.0, 0.0], [0.0, 1.0], -[1.0, 0.0], -[0.0, 1.0]]
x_dx_list = map(x -> (x, map(A -> A*x, A_list)), x_list)

## Solving
print_period = 1
δ, c_list = CLC.learn_candidate_lyapunov_function(
    method, x_dx_list,
    G0, Gmax, r0, rmin, tol_faces,
    print_period, solver)

## Plotting
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

norm_poly(x) = maximum(c -> c'*x, c_list)
Z = map(x -> norm_poly(x), Xf)
zm = min(minimum(Z[1, :]), minimum(Z[:, 1]))/1.2

ax.contour(X1, X2, Z, levels=(-1, zm), colors="gold")
h = ax.contourf(X1, X2, Z, levels=(-1, zm), colors = "none", hatches="//")
for coll in h.collections
    coll.set_edgecolor("gold")
end

α = 0.2

for x_dx in x_dx_list
    x, dx_list = x_dx
    nx = norm_poly(x)
    xs = x*zm/nx
    ax.plot(xs..., marker=".", ms=10, c="red")
    for dx in dx_list
        ys = xs + dx*(α*zm/nx)
        ax.plot((xs[1], ys[1]), (xs[2], ys[2]), c="green")
    end
end

end # TestMain