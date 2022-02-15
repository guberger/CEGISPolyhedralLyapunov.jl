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
ϵ = 1e-1
tol = -eps(1.0)
D = 2
M = 5
meth = CPL.Chebyshev()

## Tests
domain = zeros(1, D)
α = 1.0
fields = [[-α +1.0; -1.0 -α]]
sys = CPL.LinearSystem(domain, fields)
systems = (sys,)

seeds_init = (CPL.Node[],)

δ_min = 0.00005
coeffs, nodes, obj_max, flag =
    CPL.process_PLF_fixed(meth, M, D, systems, seeds_init,
                          ϵ, tol, δ_min, solver,
                          depth_max=15,
                          output_depth=1000,
                          learner_output=false)

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
F = [map(x -> (sys.fields[1]*x)[i], X) for i = 1:2]
ax.quiver(X1, X2, F..., color="gray")

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

end # module