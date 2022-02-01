module TestMain

using LinearAlgebra
using JuMP
using Gurobi
using PyPlot
include("../../src/CEGARLearningCLF.jl")
CLC = CEGARLearningCLF
include("../utils/polyhedra.jl")

## Parameters
meth_learn = CLC.LearnPolyhedralPoints{2}()
meth_verify = CLC.VerifyPolyhedralMultiple{2}()
Hs1 = [[0.0, -1.0]]
As1 = [[-0.5 +1.0; -1.0 -0.5]]
Hs2 = [[0.0, +1.0]]
As2 = [[-0.0 +1.0; -1.0 -0.0]]
Hs_list = [Hs1, Hs2]
As_list = [As1, As2]
sys = CLC.PiecewiseLinearSystem{2}(2, Hs_list, As_list)
prob = CLC.CEGARProblem(sys, meth_learn, meth_verify)
G0 = 0.1
Gmax = 10.0
r0 = 0.01
rmin = 1e-6
params = (tol_faces=1e-5, tol_deriv=-1e-5,
          print_period_1=1, print_period_2=1,
          do_trace=true)
solver = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag"=>false)

## -----------------------------------------------------------------------------
## Learner illustration
np = 10
α_list = range(0, 2π, length=np + 1)[1:np]
x_list = map(α -> [cos(α), sin(α)], α_list)
x_dx_list = CLC._initial_witnesses(sys, x_list)

δ, c_list, G, r, flag = CLC.learn_candidate_lyapunov_function(
    meth_learn, x_dx_list,
    G0, Gmax, r0, rmin, params.tol_faces,
    params.print_period_1, solver)

fig = figure(0, figsize=(8, 10))
ax = fig.add_subplot(aspect="equal")

xlims = (-2, 2)
ylims = (-2, 2)
ax.set_xlim(xlims...)
ax.set_ylim(ylims...)
ax.set_xticks(-2:1:2)
ax.set_yticks(-2:1:2)
ax.tick_params(axis="both", labelsize=15)

verts = retrieve_vertices_2d(c_list)
nv = maximum(x -> norm(x, Inf), verts)
scaling = 1.8/nv

norm_dx_max = -1.0

for x_dx in x_dx_list
    global norm_dx_max
    x, dx_list = x_dx
    nx = norm_poly(x, c_list)
    for dx in dx_list
        norm_dx_max = max(norm_dx_max, norm(dx)/nx)
    end
end

ax.plot(xlims, (0, 0), ls="--", c="black", lw=1.0)
ax.plot(0, 0, marker="x", ms=10, c="black", mew=2.5)
verts_scaled = map(x -> x*scaling, verts)
polylist = matplotlib.collections.PolyCollection([verts_scaled])
fca = matplotlib.colors.colorConverter.to_rgba("gold", alpha=0.5)
polylist.set_facecolor(fca)
polylist.set_edgecolor("gold")
polylist.set_linewidth(2.0)
ax.add_collection(polylist)

α_dx = 0.6

for x_dx in x_dx_list
    x, dx_list = x_dx
    nx = norm_poly(x, c_list)
    xs = x*scaling/nx
    ax.plot(xs..., marker=".", ms=15, c="blue")
    for dx in dx_list
        dxs = dx/norm_dx_max
        ys = xs + α_dx*dxs
        ax.plot((xs[1], ys[1]), (xs[2], ys[2]), c="green", lw=2.5)
    end
end

ax.text(1.5, +1.6, L"\mathcal{Q}(x)=1",
        horizontalalignment="center", verticalalignment="center", fontsize=20)
ax.text(1.5, -1.6, L"\mathcal{Q}(x)=2",
        horizontalalignment="center", verticalalignment="center", fontsize=20)

LH = (matplotlib.patches.Patch(fc="gold", ec="gold", lw=2.5, alpha=0.5,
        label=L"V(x)\leq1"),)
ax.legend(handles=LH, fontsize=20, loc="upper left")

fig.savefig("./examples/figures/fig_exa_illustrative_learner.png", dpi=200,
    transparent=false, bbox_inches="tight")

## -----------------------------------------------------------------------------
## Verifier illustration
np = 10
α_list = range(0, 2π, length=np + 1)[1:np]
c_list = map(α -> [cos(α), sin(α)], α_list)
obj_max, x, flag, j, q, qA = CLC.verify_candidate_lyapunov_function(
    meth_verify, sys, c_list, params.tol_faces, solver)

fig = figure(1, figsize=(8, 10))
ax = fig.add_subplot(aspect="equal")

xlims = (-2, 2)
ylims = (-2, 2)
ax.set_xlim(xlims...)
ax.set_ylim(ylims...)
ax.set_xticks(-2:1:2)
ax.set_yticks(-2:1:2)
ax.tick_params(axis="both", labelsize=15)

ngrid = 20
x1_grid = range(xlims..., length=ngrid)
x2_grid_list = (range(0, ylims[2], length=(ngrid ÷ 2) + 1),
                range(ylims[1], 0, length=(ngrid ÷ 2) + 1))
for q = 1:2
    x2_grid = x2_grid_list[q]
    X = map(x -> [x...], Iterators.product(x1_grid, x2_grid))
    X1 = map(x -> x[1], X)
    X2 = map(x -> x[2], X)
    F = [map(x -> (As_list[q][1]*x)[i], X) for i = 1:2]
    ax.quiver(X1, X2, F..., color="gray")
end

verts = retrieve_vertices_2d(c_list)
nv = maximum(x -> norm(x, Inf), verts)
scaling = 1.2/nv

ax.plot(xlims, (0, 0), ls="--", c="black", lw=1.0)
ax.plot(0, 0, marker="x", ms=10, c="black", mew=2.5)
verts_scaled = map(x -> x*scaling, verts)
polylist = matplotlib.collections.PolyCollection([verts_scaled])
fca = matplotlib.colors.colorConverter.to_rgba("gold", alpha=0.5)
polylist.set_facecolor(fca)
polylist.set_edgecolor("gold")
polylist.set_linewidth(2.0)
ax.add_collection(polylist)

α_dx = 0.6
nx = norm_poly(x, c_list)
xs = x*scaling/nx
ax.plot(xs..., marker=".", ms=15, c="black")
dx = As_list[q][qA]*x
dxs = dx/norm(dx)
ys = xs + α_dx*dxs
ax.plot((xs[1], ys[1]), (xs[2], ys[2]), c="red", lw=2.5)

ax.text(1.5, +1.6, L"\mathcal{Q}(x)=1",
        horizontalalignment="center", verticalalignment="center",
        fontsize=20, alpha=1.0, bbox=Dict(["facecolor"=>"white", "alpha"=>1.0]))
ax.text(1.5, -1.6, L"\mathcal{Q}(x)=2",
        horizontalalignment="center", verticalalignment="center",
        fontsize=20, alpha=1.0, bbox=Dict(["facecolor"=>"white", "alpha"=>1.0]))

LH = (matplotlib.patches.Patch(fc="gold", ec="gold", lw=2.5, alpha=0.5,
        label=L"V(x)\leq1"),)
ax.legend(handles=LH, fontsize=20, loc="upper left", facecolor="white", framealpha=1.0)

fig.savefig("./examples/figures/fig_exa_illustrative_verifier.png", dpi=200,
    transparent=false, bbox_inches="tight")

## -----------------------------------------------------------------------------
## Process illustration
x_list = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
c_list, x_dx_list, deriv, flag, trace = CLC.process_lyapunov_function(
    prob, x_list, G0, Gmax, r0, rmin, params, solver)

## Plotting
fig = figure(2, figsize=(8, 10))
ax_ = fig.subplots(nrows=4, ncols=3,
    gridspec_kw=Dict("wspace"=>0.1, "hspace"=>0.1),
    subplot_kw=Dict("aspect"=>"equal"))
n_trace = length(trace.c_list)
indexes1 = 1:min(6, n_trace)
indexes2 = n_trace ≤ 12 ? (7:n_trace) :
    unique(round.(Int, range(6, n_trace, length=7)[2:7]))
indexes = vcat(indexes1, indexes2)
n_plot = length(indexes)
mapplottr = reshape(1:12, 4, 3)
mapplot = Matrix(mapplottr')

xlims = (-2, 2)
ylims = (-2, 2)

for (k, ax) in enumerate(ax_)
    ax.set_xlim(xlims...)
    ax.set_ylim(ylims...)
    ax.set_xticks(())
    ax.set_yticks(())
    if k in (1, 2, 3, 4)
        ax.set_xticks
        ax.set_yticks(-2:1:2)
    end
    if k in (4, 8, 12)
        ax.set_xticks(-2:1:2)
    end
end

nv_max = -1.0
verts_list = Vector{Vector{CLC.vec_type}}(undef, n_plot)

for k = 1:n_plot
    global nv_max
    idx = indexes[k]
    c_list = trace.c_list[idx]
    verts = retrieve_vertices_2d(c_list)
    verts_list[k] = verts
    nv_max = max(nv_max, maximum(x -> norm(x, Inf), verts))
end

scaling = 1.8/nv_max

norm_dx_max = -1.0

for k = 1:n_plot
    global norm_dx_max
    idx = indexes[k]
    c_list = trace.c_list[idx]
    x_dx_list = trace.x_dx_list[idx]
    for x_dx in x_dx_list
        x, dx_list = x_dx
        nx = norm_poly(x, c_list)
        for dx in dx_list
            norm_dx_max = max(norm_dx_max, norm(dx)/nx)
        end
    end
end

α_dx = 0.6

for k = 1:n_plot
    ax = ax_[mapplot[k]]
    idx = indexes[k]
    ax.plot(xlims, (0, 0), ls="--", c="black", lw=0.5)
    ax.plot(0, 0, marker="x", ms=5, c="black", mew=1.5)
    c_list = trace.c_list[idx]
    verts = map(x -> x*scaling, verts_list[k])
    polylist = matplotlib.collections.PolyCollection([verts])
    fca = matplotlib.colors.colorConverter.to_rgba("gold", alpha=0.5)
    polylist.set_facecolor(fca)
    polylist.set_edgecolor("gold")
    polylist.set_linewidth(1.0)
    ax.add_collection(polylist)
    x_dx_list = trace.x_dx_list[idx]
    for x_dx in x_dx_list
        x, dx_list = x_dx
        nx = norm_poly(x, c_list)
        xs = x*scaling/nx
        ax.plot(xs..., marker=".", ms=7.5, c="blue")
        for dx in dx_list
            dxs = dx/norm_dx_max
            ys = xs + α_dx*dxs
            ax.plot((xs[1], ys[1]), (xs[2], ys[2]), c="green", lw=1.5)
        end
    end
    if idx ≤ n_trace - 1
        x, dx_list = trace.x_dx[idx]
        nx = norm_poly(x, c_list)
        xs = x*scaling/nx
        ax.plot(xs..., marker=".", ms=7.5, c="black")
        for dx in dx_list
            dxs = dx/norm_dx_max
            ys = xs + α_dx*dxs
            ax.plot((xs[1], ys[1]), (xs[2], ys[2]), c="red", lw=1.5)
        end
    end
    ax.text(0.0, 1.6, string("Step ", idx),
        horizontalalignment="center", fontsize=14)
end

x0 = [1.0, -1e-6]
idx = indexes[12]
c_list = trace.c_list[idx]
x = x0*scaling/norm_poly(x0, c_list)
ax = ax_[12]

ax.plot(x0..., marker=".", ms=7.5, c="purple")

nstep = 100
dt = 4π/nstep
xplot_seq = [Vector{Float64}(undef, nstep) for i = 1:2]

for t = 1:nstep
    global x
    for i = 1:2
        xplot_seq[i][t] = x[i]
    end
    q = 0
    for qbis = 1:length(Hs_list)
        if all(h -> h'*x ≤ 0, Hs_list[qbis])
            q = qbis
            break
        end
    end
    A = As_list[q][1]
    x = exp(A*dt)*x
end

ax.plot(xplot_seq[1], xplot_seq[2], lw=1.5, c="purple")

fig.savefig("./examples/figures/fig_exa_illustrative_process.png", dpi=200,
    transparent=false, bbox_inches="tight")

end # TestMain