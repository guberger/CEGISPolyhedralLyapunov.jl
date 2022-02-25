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

#=
System with four pieces. Trajectories are moving in square, except in the 1st
orthant where it gets attracted toward the center. The whole system is stable
but does not admit a polynomial Lyapunov function.

With α = 0.5 & M = 4: time ≈ 238 sec [DELL CU] (#iter ≈ 21_000)
=#

## Parameters
α = 0.5
domain1 = [0.0 -1.0; -1.0 0.0] # O1
fields1 = [[1.0-α 1.0; -1.0 -1.0-α]]
domain2 = [0.0 +1.0; -1.0 0.0] # O2
fields2 = [[-1.0 1.0; -1.0 1.0]]
domain3 = [0.0 +1.0; +1.0 0.0] # O3
fields3 = [[1.0 1.0; -1.0 -1.0]]
domain4 = [0.0 -1.0; +1.0 0.0] # O4
fields4 = [[-1.0 1.0; -1.0 1.0]]
sys1 = CPL.LinearSystem(domain1, fields1)
sys2 = CPL.LinearSystem(domain2, fields2)
sys3 = CPL.LinearSystem(domain3, fields3)
sys4 = CPL.LinearSystem(domain4, fields4)
systems = (sys1, sys2, sys3, sys4)
D = 2

## -----------------------------------------------------------------------------
## Field and trajectory

fig = figure(0, figsize=(8, 8))
ax = fig.add_subplot(aspect="equal")

xlims = (-2, 2)
ylims = (-2, 2)
ax.set_xlim(xlims...)
ax.set_ylim(ylims...)
ax.set_xticks(-2:1:2)
ax.set_yticks(-2:1:2)
ax.tick_params(axis="both", labelsize=15)

ax.plot(xlims, (0, 0), ls="--", c="black", lw=1.0)
ax.plot((0, 0), ylims, ls="--", c="black", lw=1.0)
ax.plot(0, 0, marker="x", ms=10, c="black", mew=2.5)

ngrid = 20
x1_grid = range(xlims..., length=ngrid)
x2_grid = range(ylims..., length=ngrid)
X = map(x -> [x...], Iterators.product(x1_grid, x2_grid))
X1 = map(x -> x[1], X)
X2 = map(x -> x[2], X)

for sys in systems
    F = [map(x -> NaN, X) for k = 1:2]
    for (i, x) in enumerate(X)
        if all(sys.domain * x .≤ 0)
            for k = 1:2
                F[k][i] = (sys.fields[1]*x)[k]
            end
        end
    end
    ax.quiver(X1, X2, F..., color="gray")
end

x = [1.5, -1e-6]

ax.plot(x..., marker=".", ms=15, c="purple")

nstep = 1000
dt = 8π/nstep
xplot_seq = [Vector{Float64}(undef, nstep) for i = 1:2]

for t = 1:nstep
    global x
    for i = 1:2
        xplot_seq[i][t] = x[i]
    end
    for sys in systems
        if all(sys.domain * x .≤ 0)
            A = sys.fields[1]
            x = exp(A*dt)*x
            break
        end
    end   
end

ax.plot(xplot_seq[1], xplot_seq[2], lw=1.5, c="purple")

ax.text(+1.5, +1.6, L"\mathcal{Q}(x)=1",
        horizontalalignment="center", verticalalignment="center",
        fontsize=20, backgroundcolor="white")
ax.text(+1.5, -1.6, L"\mathcal{Q}(x)=2",
        horizontalalignment="center", verticalalignment="center",
        fontsize=20, backgroundcolor="white")
ax.text(-1.5, -1.6, L"\mathcal{Q}(x)=3",
        horizontalalignment="center", verticalalignment="center",
        fontsize=20, backgroundcolor="white")
ax.text(-1.5, +1.6, L"\mathcal{Q}(x)=4",
        horizontalalignment="center", verticalalignment="center",
        fontsize=20, backgroundcolor="white")

fig.savefig("./examples/figures/fig_exa_illustrative_fixed_field.png",
            dpi=200, transparent=false, bbox_inches="tight")

## Verifier

## -----------------------------------------------------------------------------
## Verifier illustration
coeffs = [[0.75, 0], [0.75, 0.75], [0, -0.75], [-0.01, 0], [0, 0.75]]
ζ = 1e5
x = [-1, 0]

fig = figure(1, figsize=(14, 10))
ax_ = fig.subplots(nrows=1, ncols=2,
    gridspec_kw=Dict("wspace"=>0.05, "hspace"=>0.1),
    subplot_kw=Dict("aspect"=>"equal"))

ax = ax_[1]
xlims = (-2, 2)
ylims = (-2, 2)
ax.set_xlim(xlims...)
ax.set_ylim(ylims...)
ax.set_xticks(-2:1:2)
ax.set_yticks(-2:1:2)
ax.tick_params(axis="both", labelsize=15)

ax.plot(xlims, (0, 0), ls="--", c="black", lw=1.0)
ax.plot((0, 0), ylims, ls="--", c="black", lw=1.0)
ax.plot(0, 0, marker="x", ms=10, c="black", mew=2.5)
ax.plot((1, 1, -1, -1, 1), (1, -1, -1, 1, 1), ls="-", c="purple", lw=1.0)

verts = retrieve_vertices_2d(coeffs)
scaling = 1.0

ax.plot(xlims, (0, 0), ls="--", c="black", lw=1.0)
ax.plot(0, 0, marker="x", ms=10, c="black", mew=2.5)
verts_scaled = map(x -> x*scaling, verts)
polylist = matplotlib.collections.PolyCollection([verts_scaled])
fca = matplotlib.colors.colorConverter.to_rgba("gold", alpha=0.5)
polylist.set_facecolor(fca)
polylist.set_edgecolor("gold")
polylist.set_linewidth(2.0)
ax.add_collection(polylist)

ax.plot(x..., marker=".", ms=15, c="red")

ax.text(+1.5, +1.6, L"\mathcal{Q}(x)=1",
        horizontalalignment="center", verticalalignment="center",
        fontsize=20, backgroundcolor="white")
ax.text(+1.5, -1.6, L"\mathcal{Q}(x)=2",
        horizontalalignment="center", verticalalignment="center",
        fontsize=20, backgroundcolor="white")
ax.text(-1.5, -1.6, L"\mathcal{Q}(x)=3",
        horizontalalignment="center", verticalalignment="center",
        fontsize=20, backgroundcolor="white")
ax.text(-1.5, +1.6, L"\mathcal{Q}(x)=4",
        horizontalalignment="center", verticalalignment="center",
        fontsize=20, backgroundcolor="white")

LH = (matplotlib.patches.Patch(fc="gold", ec="gold", lw=2.5, alpha=0.5,
        label=L"V(x)\leq1"),)
ax.legend(handles=LH, fontsize=20, loc="upper center",
          bbox_to_anchor=(0.5, 1.15), facecolor="white", framealpha=1.0)

np = 10
α_list = range(0, 2π, length=np + 1)[1:np]
coeffs = map(α -> [cos(α), sin(α)], α_list)
ζ = 1e5
x = Vector{Float64}(undef, D)
obj_max, flag, i, q, σ = CPL.verify_PLF!(np, D, x, systems, coeffs, ζ, solver)

ax = ax_[2]

xlims = (-2, 2)
ylims = (-2, 2)
ax.set_xlim(xlims...)
ax.set_ylim(ylims...)
ax.set_xticks(-2:1:2)
ax.set_yticks(())
ax.tick_params(axis="both", labelsize=15)

ax.plot(xlims, (0, 0), ls="--", c="black", lw=1.0)
ax.plot((0, 0), ylims, ls="--", c="black", lw=1.0)
ax.plot(0, 0, marker="x", ms=10, c="black", mew=2.5)

ngrid = 20
x1_grid = range(xlims..., length=ngrid)
x2_grid = range(ylims..., length=ngrid)
X = map(x -> [x...], Iterators.product(x1_grid, x2_grid))
X1 = map(x -> x[1], X)
X2 = map(x -> x[2], X)

for sys in systems
    F = [map(x -> NaN, X) for k = 1:2]
    for (i, x) in enumerate(X)
        if all(sys.domain * x .≤ 0)
            for k = 1:2
                F[k][i] = (sys.fields[1]*x)[k]
            end
        end
    end
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
ax.plot(xs..., marker=".", ms=15, c="red")
dx = systems[q].fields[σ]*x
dxs = dx/norm(dx)
ys = xs + α_dx*dxs
ax.plot((xs[1], ys[1]), (xs[2], ys[2]), c="red", lw=2.5)

ax.text(+1.5, +1.6, L"\mathcal{Q}(x)=1",
        horizontalalignment="center", verticalalignment="center",
        fontsize=20, backgroundcolor="white")
ax.text(+1.5, -1.6, L"\mathcal{Q}(x)=2",
        horizontalalignment="center", verticalalignment="center",
        fontsize=20, backgroundcolor="white")
ax.text(-1.5, -1.6, L"\mathcal{Q}(x)=3",
        horizontalalignment="center", verticalalignment="center",
        fontsize=20, backgroundcolor="white")
ax.text(-1.5, +1.6, L"\mathcal{Q}(x)=4",
        horizontalalignment="center", verticalalignment="center",
        fontsize=20, backgroundcolor="white")

LH = (matplotlib.patches.Patch(fc="gold", ec="gold", lw=2.5, alpha=0.5,
        label=L"V(x)\leq1"),)
ax.legend(handles=LH, fontsize=20, loc="upper center",
          bbox_to_anchor=(0.5, 1.15), facecolor="white", framealpha=1.0)

fig.savefig("./examples/figures/fig_exa_illustrative_fixed_verifier.png",
            dpi=200, transparent=false, bbox_inches="tight")

error()

ϵ = 1e-2
tol = -1e-9
M = 4
meth = CPL.Chebyshev()

seeds_init = (CPL.Node[],)

δ_min = 1e-7
@time coeffs, nodes, obj_max, flag =
    CPL.process_PLF_fixed(meth, M, D, systems, seeds_init,
                          ϵ, tol, δ_min, solver,
                          depth_max=20,
                          output_period=200, level_output=0)

fig = figure(0, figsize=(8, 8))
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
x2_grid = range(ylims..., length=ngrid)
X = map(x -> [x...], Iterators.product(x1_grid, x2_grid))
X1 = map(x -> x[1], X)
X2 = map(x -> x[2], X)

for sys in systems
    F = [map(x -> NaN, X) for k = 1:2]
    for (i, x) in enumerate(X)
        if all(sys.domain * x .≤ 0)
            for k = 1:2
                F[k][i] = (sys.fields[1]*x)[k]
            end
        end
    end
    ax.quiver(X1, X2, F..., color="gray")
end

verts = retrieve_vertices_2d(coeffs)
nv = maximum(x -> norm(x, Inf), verts)
scaling = 1.8/nv

norm_dx_max = -1.0

for node in nodes
    flow = node.witness.flow
    global norm_dx_max
    nx = norm_poly(flow.point, coeffs)
    for dx in flow.grads
        norm_dx_max = max(norm_dx_max, norm(dx)/nx)
    end
end

verts_scaled = map(x -> x*scaling, verts)
polylist = matplotlib.collections.PolyCollection([verts_scaled])
fca = matplotlib.colors.colorConverter.to_rgba("gold", alpha=0.5)
polylist.set_facecolor(fca)
polylist.set_edgecolor("gold")
polylist.set_linewidth(2.0)
ax.add_collection(polylist)

α_dx = 0.6

for node in nodes
    flow = node.witness.flow
    x = flow.point
    nx = norm_poly(x, coeffs)
    xs = x*scaling/nx
    ax.plot(xs..., marker=".", ms=15, c="blue")
    for dx in flow.grads
        dxs = dx/(nx*norm_dx_max)
        ys = xs + α_dx*dxs
        ax.plot((xs[1], ys[1]), (xs[2], ys[2]), c="green", lw=2.5)
    end
end

x0 = [1.0, -1e-6]
x = x0*scaling/norm_poly(x0, coeffs)

ax.plot(x..., marker=".", ms=7.5, c="purple")

nstep = 400
dt = 2π/nstep
xplot_seq = [Vector{Float64}(undef, nstep) for i = 1:2]

for t = 1:nstep
    global x
    for i = 1:2
        xplot_seq[i][t] = x[i]
    end
    for sys in systems
        if all(sys.domain * x .≤ 0)
            A = sys.fields[1]
            x = exp(A*dt)*x
            break
        end
    end   
end

ax.plot(xplot_seq[1], xplot_seq[2], lw=1.5, c="purple")

fig.savefig("./examples/figures/fig_exa_illustrative_fixed_square.png",
            dpi=200, transparent=false, bbox_inches="tight")

end # module