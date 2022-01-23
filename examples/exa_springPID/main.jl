module TestMain

using LinearAlgebra
using PyPlot
using PyCall
art3d = PyObject(PyPlot.art3D)
include("../utils/polyhedra.jl")

str = readlines("./examples/exa_springPID/lyapunov-470-150-12.txt")
M = length(str)
c_list = Vector{Vector{Float64}}(undef, M)

for (j, ln) in enumerate(str)
    ln = replace(ln, r"[\[\],]"=>"")
    words = split(ln)
    @assert length(words) == 3
    c_list[j] = parse.(Float64, words)
end

simplices = retrieve_simplices_3d(c_list)

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
ax.set_ylim((-12, 10))
ax.set_zlim((-50, 50))
ax.set_xticks(-2:1:2)
ax.set_yticks(-10:5:10)
ax.set_zticks(-40:20:40)
ax.tick_params(axis="both", which="major", labelsize=15)

ax.yaxis.set_rotate_label(false)
ax.zaxis.set_rotate_label(false)
ax.set_xlabel("x", fontsize=28)
ax.set_ylabel(L"\dot{x}", fontsize=28, rotation="horizontal", labelpad=9)
ax.set_zlabel(L"\ddot{x}", fontsize=28, rotation="horizontal")

ax.view_init(elev=10.0, azim=-135)

fig.savefig("./examples/figures/fig_exa_mass_spring_lyapunov.png", dpi=200,
    transparent=false,
    bbox_inches=matplotlib.transforms.Bbox(((1.05, 1.80), (7.15, 7.6))))

x = [0.0, 1.0, 0.0]
dt = 0.001
tmax = 1.0
tspan = 0.0:dt:tmax
nstep = length(tspan)
Vplot_seq = Vector{Float64}(undef, nstep)
a1 = 10
a0 = 20
Ki = 470
Kp = 150
Kd = 12

for t = 1:nstep
    global x
    Vplot_seq[t] = norm_poly(x, c_list)
    xnext = Vector{Float64}(undef, 3)
    if Kd*x[3] + Kp*x[2] + Ki*x[1] ≥ 0
        xnext[1] = x[1] + dt*(x[2])
        xnext[2] = x[2] + dt*(x[3])
        xnext[3] = x[3] + dt*(-Ki*x[1] - (a0 + Kp)*x[2] - (a1 + Kd)*x[3])
    else
        xnext[1] = x[1] + dt*(-10*x[1] + x[2])
        xnext[2] = x[2] + dt*(x[3])
        xnext[3] = x[3] + dt*(-a0*x[2] - a1*x[3])
    end
    x = xnext
end

fig = figure(1, figsize=(10, 3))
ax = fig.add_subplot()

ax.plot(tspan, Vplot_seq, lw=2.5)
ax.set_xlabel("t", fontsize=15)
ax.set_ylabel("V(x(t))", fontsize=15)
ax.tick_params(axis="both", which="major", labelsize=12)

fig.savefig("./examples/figures/fig_exa_mass_spring_decrease.png", dpi=200,
    transparent=false, bbox_inches="tight")

end # TestMain