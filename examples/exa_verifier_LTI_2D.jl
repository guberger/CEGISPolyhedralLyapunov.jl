module TestMain

using Random
using JuMP
using Gurobi
using PyPlot
include("../src/CEGARLearningCLF.jl")
CLC = CEGARLearningCLF

Random.seed!(0)

## Parameters
method_s = CLC.VerifyPolyhedralSingle{2}()
method_m = CLC.VerifyPolyhedralMultiple{2}()
A = [0.0 0.0; 1.0 0.0]
A_list = [A]
solver = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag"=>false)

N = 5
α_list = range(0.0, stop=2π, length=N)
c_list = map(α -> [cos(α), sin(α)], α_list)

## Solving
obj_max, x = @time CLC.verify_candidate_lyapunov_function(method_s, A_list,
                                                    c_list, solver)
obj_max, x = @time CLC.verify_candidate_lyapunov_function(method_m, A_list,
                                                    c_list, solver)

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

norm_poly(x) = maximum(c -> abs(c'*x), c_list)
Z = map(x -> norm_poly(x), Xf)
zm = min(minimum(Z[1, :]), minimum(Z[:, 1]))/1.2

ax.contour(X1, X2, Z, levels=(-1, zm), colors="gold")
h = ax.contourf(X1, X2, Z, levels=(-1, zm), colors = "none", hatches="//")
for coll in h.collections
    coll.set_edgecolor("gold")
end

α = 0.2
x_dx_list = [(x, map(A -> A*x, A_list))]

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