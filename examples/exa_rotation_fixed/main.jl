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
Simple rotation + attraction system with different levels of attrachtions
depeding on the region of the state space.

With α1 = 0.5 & α2 = 0.5 & M = 8: time ≈ 154 sec [DELL CU] (#iter ≈ 22_000)
=#

## Parameters
α1 = 0.5
domain1 = [0.0 -1.0]
fields1 = [[-α1 +1.0; -1.0 -α1]]
α2 = 0.75
domain2 = [0.0 +1.0]
fields2 = [[-α2 +1.0; -1.0 -α2]]
sys1 = CPL.LinearSystem(domain1, fields1)
sys2 = CPL.LinearSystem(domain2, fields2)
systems = (sys1, sys2)
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

np = 50
α_list = range(0, 2π, length=np + 1)[1:np]
coeffs = map(α -> [cos(α), sin(α)], α_list)
verts = retrieve_vertices_2d(coeffs)
scaling = 1.5

ax.plot(xlims, (0, 0), ls="--", c="black", lw=1.0)
ax.plot(0, 0, marker="x", ms=10, c="black", mew=2.5)
verts_scaled = map(x -> x*scaling, verts)
polylist = matplotlib.collections.PolyCollection([verts_scaled])
fca = matplotlib.colors.colorConverter.to_rgba("gold", alpha=0.5)
polylist.set_facecolor(fca)
polylist.set_edgecolor("gold")
polylist.set_linewidth(2.0)
ax.add_collection(polylist)

x0 = [1.0, -1e-6]
x = x0*scaling/norm_poly(x0, coeffs)

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

fig.savefig("./examples/figures/fig_exa_rotation_fixed_field.png",
            dpi=200, transparent=false, bbox_inches="tight")

error()

ϵ = 1e-2
tol = -1e-9
M = 8
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

fig.savefig("./examples/figures/fig_exa_illustrative_fixed_rotate.png",
            dpi=200, transparent=false, bbox_inches="tight")

end # module