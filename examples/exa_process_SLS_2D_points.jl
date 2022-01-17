module TestMain

using LinearAlgebra
using Random
using JuMP
using Gurobi
using PyPlot
include("../src/CEGARLearningCLF.jl")
CLC = CEGARLearningCLF

Random.seed!(0)

## Parameters
meth_learn = CLC.LearnPolyhedralPoints{2}()
meth_verify = CLC.VerifyPolyhedralMultiple{2}()
A1 = [-0.2 1.0; -1.0 -0.2]
A2 = [-0.2 0.0; -0.5 -0.2]
# A1 = [-0.5 1.0; -1.0 -0.5]
# A2 = [-0.2 0.0; -0.5 -0.2]
A_list = [A1, A2]
prob = CLC.CEGARProblem{2}(A_list, meth_learn, meth_verify)
G0 = 0.1
Gmax = 10.0
r0 = 0.01
rmin = 1e-6
params = (tol_faces=1e-5, tol_deriv=1e-5,
          print_period_1=1, print_period_2=1,
          do_trace=true)
solver = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag"=>false)

N = 4
angles = range(0.0, 2π, length=N + 1)[1:N]
x_list = [[cos(θ), sin(θ)] for θ in angles]

## Solving
c_list, x_dx_list, deriv, flag, trace = CLC.process_lyapunov_function(
    prob, x_list, G0, Gmax, r0, rmin, params, solver)

## Plotting
# matplotlib.rc("legend", fontsize=25)
# matplotlib.rc("axes", labelsize=20)
# matplotlib.rc("xtick", labelsize=20)
# matplotlib.rc("ytick", labelsize=20)
# matplotlib.rc("text", usetex=true)

fig = figure(0, figsize=(8, 10))
ax_ = fig.subplots(nrows=4, ncols=3,
    gridspec_kw=Dict("wspace"=>0.1, "hspace"=>0.1),
    subplot_kw=Dict("aspect"=>"equal"))
xlims = (-2, 2)
ylims = (-2, 2)

for (k, ax) in enumerate(ax_)
    ax.set_xlim(xlims...)
    ax.set_ylim(ylims...)
    ax.set_xticklabels(())
    ax.set_yticklabels(())
    if k in (1, 2, 3, 4)
        ax.set_yticklabels(-2:0.5:2)
    end
    if k in (4, 8, 12)
        ax.set_xticklabels(-2:0.5:2)
    end
end

np = 600
xf1 = range(xlims[1], stop = xlims[2], length = np)
xf2 = range(ylims[1], stop = ylims[2], length = np)
Xftemp = collect(Iterators.product(xf1, xf2))
Xf = map(x -> [x...], Xftemp)
X1 = map(x -> x[1], Xf)
X2 = map(x -> x[2], Xf)

n_trace = length(trace.c_list)
indexes1 = 1:min(6, n_trace)
indexes2 = n_trace ≤ 12 ? (7:n_trace) :
    unique(round.(Int, range(6, n_trace, length=7)[2:7]))
indexes = vcat(indexes1, indexes2)
n_plot = length(indexes)
Z_ = Vector{Matrix{Float64}}(undef, n_plot)
zm = Inf
norm_poly(c_list) = x -> maximum(c -> abs(c'*x), c_list)

for k = 1:n_plot
    global zm
    idx = indexes[k]
    c_list = trace.c_list[idx]
    fn = norm_poly(c_list)
    Z = map(x -> fn(x), Xf)
    zm = min(zm, minimum(Z[1, :]), minimum(Z[:, 1]))
    Z_[k] = Z
end

zm = zm/1.2
α = 0.3
map1 = reshape(1:12, 4, 3)
map2 = Matrix(map1')

for k = 1:n_plot
    ax = ax_[map2[k]]
    idx = indexes[k]
    Z = Z_[k]
    c_list = trace.c_list[idx]
    fn = norm_poly(c_list)
    x_dx_list = trace.x_dx_list[idx]
    ax.contour(X1, X2, Z, levels=(-1, zm), colors="gold")
    h = ax.contourf(X1, X2, Z, levels=(-1, zm), colors = "gold", alpha=0.5)
    for x_dx in x_dx_list
        x, dx_list = x_dx
        nx = fn(x)
        xs = x*zm/nx
        ax.plot(xs..., marker=".", ms=7.5, c="blue")
        for dx in dx_list
            dxs = dx*zm/norm(dx)
            ys = xs + α*dxs
            ax.plot((xs[1], ys[1]), (xs[2], ys[2]), c="green", lw=1.5)
        end
    end
    if idx ≤ n_trace - 1
        x, dx_list = trace.x_dx[idx]
        nx = fn(x)
        xs = x*zm/nx
        ax.plot(xs..., marker=".", ms=7.5, c="black")
        for dx in dx_list
            dxs = dx*zm/norm(dx)
            ys = xs + α*dxs
            ax.plot((xs[1], ys[1]), (xs[2], ys[2]), c="red", lw=1.5)
        end
    end
    ax.text(0.0, 1.6, string("Step ", idx),
        horizontalalignment="center", fontsize=14)
end

# LH = (
#     matplotlib.lines.Line2D([0], [0], c="red", ls="none", marker=".", ms=20,
#         label=L"\hat{x}_i"),
#     matplotlib.lines.Line2D([0], [0], c="green", lw=2,
#         label=L"A_q\hat{x}_i"),
#     matplotlib.patches.Patch(fc="none", ec="gold", hatch="//",
#         label=L"\{V(x)\leq c\}")
#     )
# ax.legend(handles=LH, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))

fig.savefig("./figures/fig_process_example.png", dpi=200,
    transparent=false, bbox_inches="tight")

end # TestMain