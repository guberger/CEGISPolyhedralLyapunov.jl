module ExampleSpringPID

using LinearAlgebra
using PyPlot
using PyCall
art3d = PyObject(PyPlot.art3D)
include("../../../src/CEGISPolyhedralLyapunov.jl")
CPLP = CEGISPolyhedralLyapunov.Polyhedra
include("../../utils/geometry.jl")

datafile = "dataset_3"
include(string("./datasets/", datafile, ".jl"))

str = readlines(string(@__DIR__, "/results/", datafile, ".txt"))
M = length(str)
vecs = Vector{Vector{Float64}}(undef, M)

for (i, ln) in enumerate(str)
    ln = replace(ln, r"[\[\],]"=>"")
    words = split(ln)
    @assert length(words) == 3
    vecs[i] = parse.(Float64, words)
end

p = CPLP.Polyhedron()
for vec in vecs
    CPLP.add_halfspace!(p, CPLP.Halfspace(vec, -1.0))
end
simplices = compute_simplices_3d(p)

fig = figure(0, figsize=(8, 10))
ax = fig.add_subplot(projection="3d")
ax.xaxis.pane.fill = false
ax.yaxis.pane.fill = false
ax.zaxis.pane.fill = false

polylist = art3d.Poly3DCollection(simplices)
fca = matplotlib.colors.colorConverter.to_rgba("gold", alpha=0.5)
polylist.set_facecolor(fca)
polylist.set_edgecolor("black")
polylist.set_linewidth(1.0)
ax.add_collection3d(polylist)

ax.set_xlim((-2, 2))
ax.set_ylim((-8, 8))
ax.set_zlim((-35, 65))
ax.set_xticks(-2:1:2)
ax.set_yticks(-8:4:8)
ax.set_zticks(-30:15:60)
ax.tick_params(axis="both", which="major", labelsize=15)

ax.yaxis.set_rotate_label(false)
ax.zaxis.set_rotate_label(false)
ax.set_xlabel(L"x", fontsize=28, usetex=true)
ax.set_ylabel(L"\dot{x}", fontsize=28, rotation="horizontal", labelpad=9, usetex=true)
ax.set_zlabel(L"\ddot{x}", fontsize=28, rotation="horizontal", usetex=true)

ax.view_init(elev=10.0, azim=-135)

filename = 
fig.savefig(string(
        @__DIR__,
        "/../../figures/AdaptiveComplexity/fig_exa_mass_spring_lyapunov.png"
    ),
    dpi=200,
    transparent=false,
    bbox_inches=matplotlib.transforms.Bbox(((1.05, 1.80), (7.15, 7.6)))
)

x = [0.0, 1.0, 0.0]
dt = 0.001
tmax = 1.0
tspan = 0.0:dt:tmax
nstep = length(tspan)
Vplot_seq = Vector{Float64}(undef, nstep)

for t = 1:nstep
    global x
    Vplot_seq[t] = maximum(vec -> dot(vec, x), vecs)
    xnext = Vector{Float64}(undef, 3)
    if Kd*x[3] + Kp*x[2] + Ki*x[1] ≥ 0
        xnext[1] = x[1] + dt*(x[2])
        xnext[2] = x[2] + dt*(x[3])
        xnext[3] = x[3] + dt*(-Ki*x[1] - (a0 + Kp)*x[2] - (a1 + Kd)*x[3])
    else
        xnext[1] = x[1] + dt*(-c_aw*x[1] + x[2])
        xnext[2] = x[2] + dt*(x[3])
        xnext[3] = x[3] + dt*(-a0*x[2] - a1*x[3])
    end
    x = xnext
end

fig = figure(1, figsize=(10, 3))
ax = fig.add_subplot()

ax.plot(tspan, Vplot_seq, lw=2.5)
ax.set_xlabel(L"t", fontsize=23, usetex=true)
ax.set_ylabel(L"V(x(t))", fontsize=23, usetex=true)
ax.tick_params(axis="both", which="major", labelsize=12)

fig.savefig(string(
        @__DIR__,
        "/../../figures/AdaptiveComplexity/fig_exa_mass_spring_decrease.png"
    ),
    dpi=200, transparent=false, bbox_inches="tight")

end # module