module ExampleSpringPID_Plot

using LinearAlgebra
using PyPlot
include("../utils/plotting.jl")

datafile = "dataset_3"
include(string("./datasets/", datafile, ".jl"))

str = readlines(string(@__DIR__, "/results/", datafile, ".txt"))
lfs = Vector{Float64}[]

for ln in str
    local ln = replace(ln, r"[\[\],]"=>"")
    local words = split(ln)
    @assert length(words) == 3
    push!(lfs, parse.(Float64, words))
end

lims = ((-100, -100, -100), (100, 100, 100))
fig = figure(0, figsize=(8, 12))
ax = fig.add_subplot(projection="3d")
ax.set_box_aspect((4, 4, 5))
ax.xaxis.pane.fill = false
ax.yaxis.pane.fill = false
ax.zaxis.pane.fill = false

plot_level3D!(ax, lfs, 1, lims, fc="gold", ec="black", ew=1)

ax.set_xlim((-2, 2))
ax.set_ylim((-6, 6))
ax.set_zlim((-20, 25))
ax.set_xticks(-2:1:2)
ax.set_yticks(-6:3:6)
ax.set_zticks(-20:10:20)
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
    bbox_inches=matplotlib.transforms.Bbox(((1.1, 2.9), (6.75, 8.5)))
)

end # module