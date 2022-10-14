module ExampleZelentsowsky_Plot

using LinearAlgebra
using PyPlot
include("../utils/plotting.jl")

datafile = "dataset_1"
include(string("./datasets/", datafile, ".jl"))

str = readlines(string(@__DIR__, "/results/", datafile, ".txt"))
lfs = Vector{Float64}[]

for ln in str
    local ln = replace(ln, r"[\[\],]"=>"")
    local words = split(ln)
    @assert length(words) == 2
    push!(lfs, parse.(Float64, words))
end

A = [0.0 1.0; -2.0 -1.0]
B = [0.0 0.0; -1.0 0.0]

Ms = (A, A + α*B)

fig = figure(0, figsize=(8, 10))
ax = fig.add_subplot(aspect="equal")

xlims = (-3, 3)
ylims = (-3, 3)
ax.set_xlim(xlims...)
ax.set_ylim(ylims...)
ax.set_xticks(-3:1:3)
ax.set_yticks(-3:1:3)
ax.tick_params(axis="both", labelsize=15)
ax.plot(0, 0, marker="x", ms=10, c="black", mew=2.5)

ngrid = 20
x1_grid = range(xlims..., length=ngrid)
x2_grid = range(ylims..., length=ngrid)
X = map(x -> [x...], Iterators.product(x1_grid, x2_grid))
X1 = getindex.(X, 1)
X2 = getindex.(X, 2)

for M in Ms
    local D = map(x -> M*x, X)
    local D1 = getindex.(D, 1)
    local D2 = getindex.(D, 2)
    ax.quiver(X1, X2, D1, D2, color="gray")
end

β = 1
lims = ((-10, -10), (10, 10))

plot_level2D!(ax, lfs, β, lims, fc="gold", ec="gold")

LH = (matplotlib.patches.Patch(fc="gold", ec="gold", lw=2.5, alpha=0.5,
        label=L"V(x)\leq1"),)
ax.legend(handles=LH, fontsize=20, loc="upper left",
          facecolor="white", framealpha=1.0)

fig.savefig(string(
        @__DIR__, "/../figures/fig_exa_zelentsowsky.png"
    ),
    dpi=200, transparent=false, bbox_inches="tight")

end # module