module ExampleSpringPID_Plot

using LinearAlgebra
using PyPlot
using PyCall
art3d = PyObject(PyPlot.art3D)
include("../../src/CEGISPolyhedralLyapunov.jl")
CPL = CEGISPolyhedralLyapunov
include("../utils/geometry.jl")

datafile = "dataset_3"
include(string("./datasets/", datafile, ".jl"))

str = readlines(string(@__DIR__, "/results/", datafile, ".txt"))
M = length(str)
lfs = Vector{CPL.LinForm}(undef, M)

for (i, ln) in enumerate(str)
    ln = replace(ln, r"[\[\],]"=>"")
    words = split(ln)
    @assert length(words) == 3
    lfs[i] = CPL.LinForm(parse.(Float64, words))
end

p = CPL.Polyhedron()
for lf in lfs
    CPL.add_halfspace!(p, lf.lin, -1.0)
end
simplices = compute_simplices_3d(p, zeros(3))

fig = figure(0, figsize=(8, 12))
ax = fig.add_subplot(projection="3d")
ax.set_box_aspect((4, 4, 5))
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
ax.set_zlim((-35, 35))
ax.set_xticks(-2:1:2)
ax.set_yticks(-8:4:8)
ax.set_zticks(-30:15:30)
ax.tick_params(axis="both", which="major", labelsize=15)

ax.yaxis.set_rotate_label(false)
ax.zaxis.set_rotate_label(false)
ax.set_xlabel(L"x", fontsize=28, usetex=true)
ax.set_ylabel(L"\dot{x}", fontsize=28, rotation="horizontal", labelpad=9, usetex=true)
ax.set_zlabel(L"\ddot{x}", fontsize=28, rotation="horizontal", usetex=true)

ax.view_init(elev=10.0, azim=-135)

fig.savefig(string(
        @__DIR__, "/../figures/fig_exa_mass_spring_lyapunov.png"
    ),
    dpi=200,
    transparent=false,
    bbox_inches=matplotlib.transforms.Bbox(((1.1, 0.95), (6.75, 6.5)))
)

x = [0.0, 1.0, 0.0]
dt = 0.001
tmax = 1.0
tspan = 0.0:dt:tmax
nstep = length(tspan)
Vplot_seq = Vector{Float64}(undef, nstep)

for t = 1:nstep
    global x
    Vplot_seq[t] = maximum(lf -> CPL._eval(lf, x), lfs)
    xnext = Vector{Float64}(undef, 3)
    if Kd*x[3] + Kp*x[2] + Ki*x[1] ??? 0
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
        @__DIR__, "/../figures/fig_exa_mass_spring_decrease.png"
    ),
    dpi=200, transparent=false,
    bbox_inches=matplotlib.transforms.Bbox(((0.45, -0.3), (9.1, 2.7)))
)

end # module