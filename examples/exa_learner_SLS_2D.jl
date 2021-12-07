module TestMain

using Random
using JuMP
using Gurobi
using PyPlot
include("../src/CEGARLearningCLF.jl")
CLC = CEGARLearningCLF

Random.seed!(0)

## Parameters
method = CLC.LearnPolyhedralPoints{2}()
A1 = [-0.5 1.0; -1.0 -0.5]
A2 = [-0.3 0.0; -0.5 -0.3]
A_list = [A1, A2]
G0 = 0.1
Gmax = 2
tol_faces = 1e-5
solver = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag"=>false)

N = 20
x_list = [randn(2) for i = 1:N]
x_dx_list = map(x -> (x, map(A -> A*x, A_list)), x_list)

## Solving
print_period = 1
r, c_list = CLC.learn_candidate_lyapunov_function(method, x_dx_list,
                                                  G0, Gmax, tol_faces,
                                                  print_period, solver)

## Plotting
matplotlib.rc("legend", fontsize=25)
matplotlib.rc("axes", labelsize=20)
matplotlib.rc("xtick", labelsize=20)
matplotlib.rc("ytick", labelsize=20)
matplotlib.rc("text", usetex=true)

fig = figure(0, figsize=(12, 10))
ax = fig.add_subplot(aspect="equal")
xlims = (-2, 2)
ylims = (-2, 2)
ax.set_xlim(xlims...)
ax.set_ylim(ylims...)

np = 600
xf1 = range(xlims[1], stop=xlims[2], length=np)
xf2 = range(ylims[1], stop=ylims[2], length=np)
Xftemp = collect(Iterators.product(xf1, xf2))
Xf = map(x -> [x...], Xftemp)
X1 = map(x -> x[1], Xf)
X2 = map(x -> x[2], Xf)

norm_poly(x) = maximum(c -> abs(c'*x), c_list)
Z = map(x -> norm_poly(x), Xf)
zm = min(minimum(Z[1, :]), minimum(Z[:, 1]))/1.2

ax.contour(X1, X2, Z, levels=(-1, zm), colors="gold")
h = ax.contourf(X1, X2, Z, levels=(-1, zm), colors="none", hatches="//")
for coll in h.collections
    coll.set_edgecolor("gold")
end

α = 0.2

for x_dx in x_dx_list
    x, dx_list = x_dx
    nx = norm_poly(x)
    xs = x*zm/nx
    ax.plot(xs..., marker=".", ms=15, c="red")
    for dx in dx_list
        ys = xs + dx*(α*zm/nx)
        ax.plot((xs[1], ys[1]), (xs[2], ys[2]), c="green", lw=2)
    end
end

LH = (
    matplotlib.lines.Line2D([0], [0], c="red", ls="none", marker=".", ms=20,
        label=L"\hat{x}_i"),
    matplotlib.lines.Line2D([0], [0], c="green", lw=2,
        label=L"A_q\hat{x}_i"),
    matplotlib.patches.Patch(fc="none", ec="gold", hatch="//",
        label=L"\{V(x)\leq c\}")
    )
ax.legend(handles=LH, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))

fig.savefig("./figures/fig_learner_example.png", dpi=200,
    transparent=false, bbox_inches="tight")

end # TestMain