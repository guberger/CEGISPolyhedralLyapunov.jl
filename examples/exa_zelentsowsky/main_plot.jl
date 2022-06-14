module ExampleZelentsowsky_Compute

using LinearAlgebra
using PyPlot
using PyCall
art3d = PyObject(PyPlot.art3D)
include("../../src/CEGISPolyhedralLyapunov.jl")
CPL = CEGISPolyhedralLyapunov
include("../utils/geometry.jl")

datafile = "dataset_1"
include(string("./datasets/", datafile, ".jl"))

str = readlines(string(@__DIR__, "/results/", datafile, ".txt"))
M = length(str)
lfs = Vector{CPL.LinForm}(undef, M)

for (i, ln) in enumerate(str)
    ln = replace(ln, r"[\[\],]"=>"")
    words = split(ln)
    @assert length(words) == 2
    lfs[i] = CPL.LinForm(parse.(Float64, words))
end

sys = CPL.System()

domain = CPL.Cone()
A = [0.0 1.0; -2.0 -1.0]
CPL.add_piece!(sys, domain, A)

domain = CPL.Cone()
B = [0.0 0.0; -1.0 0.0]
CPL.add_piece!(sys, domain, A + α*B)

radmax = 0.9

fig = figure(0, figsize=(8, 10))
ax = fig.add_subplot(aspect="equal")

xlims = (-1, 1)
ylims = (-1, 1)
ax.set_xlim(xlims...)
ax.set_ylim(ylims...)
ax.set_xticks(-1:0.5:1)
ax.set_yticks(-1:0.5:1)
ax.tick_params(axis="both", labelsize=15)
ax.plot(0, 0, marker="x", ms=10, c="black", mew=2.5)

ngrid = 20
x1_grid = range(xlims..., length=ngrid)
x2_grid = range(ylims..., length=ngrid)
X = map(x -> [x...], Iterators.product(x1_grid, x2_grid))
X1 = getindex.(X, 1)
X2 = getindex.(X, 2)

for piece in sys.pieces
    D = Matrix{Vector{Float64}}(undef, ngrid, ngrid)
    for (k, x) in enumerate(X)
        if x ∈ piece.domain
            D[k] = piece.A*x
        else
            D[k] = [NaN, NaN]
        end
    end
    D1 = getindex.(D, 1)
    D2 = getindex.(D, 2)
    ax.quiver(X1, X2, D1, D2, color="gray")
end

p = CPL.Polyhedron()
for lf in lfs
    CPL.add_halfspace!(p, lf.lin, -1)
end
verts = compute_vertices_2d(p, zeros(2))
verts_radius = maximum(vert -> norm(vert, Inf), verts)
scaling = radmax/verts_radius
verts_scaled = map(vert -> vert*scaling, verts)
polylist = matplotlib.collections.PolyCollection([verts_scaled])
fca = matplotlib.colors.colorConverter.to_rgba("gold", alpha=0.5)
polylist.set_facecolor(fca)
polylist.set_edgecolor("gold")
polylist.set_linewidth(2.0)
ax.add_collection(polylist)

LH = (matplotlib.patches.Patch(fc="gold", ec="gold", lw=2.5, alpha=0.5,
        label=L"V(x)\leq1"),)
ax.legend(handles=LH, fontsize=20, loc="upper left",
          facecolor="white", framealpha=1.0)

fig.savefig(string(
        @__DIR__, "/../figures/fig_exa_zelentsowsky.png"
    ),
    dpi=200, transparent=false, bbox_inches="tight")

end # module