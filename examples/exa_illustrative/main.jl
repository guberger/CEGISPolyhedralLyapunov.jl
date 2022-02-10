module ExampleIllustrative

using LinearAlgebra
using JuMP
using Gurobi
using PyPlot
include("../../src/CEGPolyhedralLyapunov.jl")
CPL = CEGPolyhedralLyapunov
include("../utils/polyhedra.jl")

const GUROBI_ENV = Gurobi.Env()
solver = optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV),
    "OutputFlag"=>false)

## Parameters
domain1 = [0.0 -1.0]
fields1 = [[-0.5 +1.0; -1.0 -0.5]]
domain2 = [0.0 +1.0]
fields2 = [[-0.0 +1.0; -1.0 -0.0]]
sys1 = CPL.LinearSystem(domain1, fields1)
sys2 = CPL.LinearSystem(domain2, fields2)
systems = (sys1, sys2)
D = 2

G0 = 0.1
Gmax = 10.0
r0 = 0.01
rmin = 1e-6
ϵ = 1e-5
tol = -1e-5

## -----------------------------------------------------------------------------
## Learner illustration
np = 10
α_list = range(0, 2π, length=np + 1)[1:np]
points = map(α -> [cos(α), sin(α)], α_list)
witnesses = CPL.make_witnesses(systems, points)

M = length(witnesses)
δ, coeffs, G, r, flag = CPL.learn_PLF_params(M, D, witnesses,
                                             G0, Gmax, r0, rmin, ϵ, solver)

fig = figure(0, figsize=(8, 10))
ax = fig.add_subplot(aspect="equal")

xlims = (-2, 2)
ylims = (-2, 2)
ax.set_xlim(xlims...)
ax.set_ylim(ylims...)
ax.set_xticks(-2:1:2)
ax.set_yticks(-2:1:2)
ax.tick_params(axis="both", labelsize=15)

verts = retrieve_vertices_2d(coeffs)
nv = maximum(x -> norm(x, Inf), verts)
scaling = 1.8/nv

norm_dx_max = -1.0

for witness in witnesses
    global norm_dx_max
    nx = norm_poly(witness.point, coeffs)
    for dx in witness.flows
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

for witness in witnesses
    x = witness.point
    nx = norm_poly(x, coeffs)
    xs = x*scaling/nx
    ax.plot(xs..., marker=".", ms=15, c="blue")
    for dx in witness.flows
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
coeffs = map(α -> [cos(α), sin(α)], α_list)
obj_max, x, flag, i, q, σ = CPL.verify_PLF(D, systems, coeffs, ϵ, solver)

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
    F = [map(x -> (systems[q].fields[1]*x)[i], X) for i = 1:2]
    ax.quiver(X1, X2, F..., color="gray")
end

verts = retrieve_vertices_2d(coeffs)
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
nx = norm_poly(x, coeffs)
xs = x*scaling/nx
ax.plot(xs..., marker=".", ms=15, c="black")
dx = systems[q].fields[σ]*x
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
ax.legend(handles=LH, fontsize=20, loc="upper left",
          facecolor="white", framealpha=1.0)

fig.savefig("./examples/figures/fig_exa_illustrative_verifier.png", dpi=200,
            transparent=false, bbox_inches="tight")

## -----------------------------------------------------------------------------
## Process illustration
points = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
witnesses_init = CPL.make_witnesses(systems, points)
coeffs, witnesses, deriv, flag, trace =
    CPL.process_PLF(D, systems, witnesses_init,
                    G0, Gmax, r0, rmin, ϵ, tol,
                    solver)

## Plotting
fig = figure(2, figsize=(8, 10))
ax_ = fig.subplots(nrows=4, ncols=3,
    gridspec_kw=Dict("wspace"=>0.1, "hspace"=>0.1),
    subplot_kw=Dict("aspect"=>"equal"))
n_trace = length(trace.coeffs_list)
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
verts_list = Vector{Vector{Vector{Float64}}}(undef, n_plot)

for k = 1:n_plot
    global nv_max
    idx = indexes[k]
    coeffs = trace.coeffs_list[idx]
    verts = retrieve_vertices_2d(coeffs)
    verts_list[k] = verts
    nv_max = max(nv_max, maximum(x -> norm(x, Inf), verts))
end

scaling = 1.8/nv_max

norm_dx_max = -1.0

for k = 1:n_plot
    global norm_dx_max
    idx = indexes[k]
    for witness in trace.witnesses_list[idx]
        nx = norm_poly(witness.point, trace.coeffs_list[idx])
        for dx in witness.flows
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
    coeffs = trace.coeffs_list[idx]
    verts = map(x -> x*scaling, verts_list[k])
    polylist = matplotlib.collections.PolyCollection([verts])
    fca = matplotlib.colors.colorConverter.to_rgba("gold", alpha=0.5)
    polylist.set_facecolor(fca)
    polylist.set_edgecolor("gold")
    polylist.set_linewidth(1.0)
    ax.add_collection(polylist)
    for witness in trace.witnesses_list[idx]
        x = witness.point
        nx = norm_poly(x, coeffs)
        xs = x*scaling/nx
        ax.plot(xs..., marker=".", ms=7.5, c="blue")
        for dx in witness.flows
            dxs = dx/norm_dx_max
            ys = xs + α_dx*dxs
            ax.plot((xs[1], ys[1]), (xs[2], ys[2]), c="green", lw=1.5)
        end
    end
    if idx ≤ n_trace - 1
        counterexample = trace.counterexample_list[idx]
        x = counterexample.point
        nx = norm_poly(x, coeffs)
        xs = x*scaling/nx
        ax.plot(xs..., marker=".", ms=7.5, c="black")
        for dx in counterexample.flows
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
x = x0*scaling/norm_poly(x0, trace.coeffs_list[idx])
ax = ax_[12]

ax.plot(x..., marker=".", ms=7.5, c="purple")

nstep = 100
dt = 4π/nstep
xplot_seq = [Vector{Float64}(undef, nstep) for i = 1:2]

for t = 1:nstep
    global x
    for i = 1:2
        xplot_seq[i][t] = x[i]
    end
    q = 0
    for sys in systems
        if all(sys.domain * x .≤ 0)
            A = sys.fields[1]
            x = exp(A*dt)*x
            break
        end
    end   
end

ax.plot(xplot_seq[1], xplot_seq[2], lw=1.5, c="purple")

fig.savefig("./examples/figures/fig_exa_illustrative_process.png", dpi=200,
            transparent=false, bbox_inches="tight")

end # module