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

ϵ = 1e-2
tol = -1e-9
M = 4
meth = CPL.Chebyshev()

seeds_init = (CPL.Node[],)

δ_min = 1e-7
coeffs, nodes, obj_max, flag =
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

for q = 1:1
    X = map(x -> [x...], Iterators.product(x1_grid, x2_grid))
    X1 = map(x -> x[1], X)
    X2 = map(x -> x[2], X)
    F = [map(x -> (systems[q].fields[1]*x)[i], X) for i = 1:2]
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

end # module