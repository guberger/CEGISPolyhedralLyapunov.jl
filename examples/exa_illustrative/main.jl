module ExampleIllustrative

using LinearAlgebra
using JuMP
using Gurobi
using PyPlot

include("../../src/CEGISPolyhedralLyapunov.jl")
CPL = CEGISPolyhedralLyapunov

include("plotting.jl")
include("../utils/plotting.jl")

const GUROBI_ENV = Gurobi.Env()
solver() = Model(optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
))

## Parameters
τ = 1/4 # 0.1

pieces = [
    CPL.Piece([-0.5 1.0; -1.0 -0.5], [[0.0, -1.0]]),
    CPL.Piece([0.01 1.0; -1.0 0.01], [[0.0, 1.0]])
]

lfs_init = [[0.34, 0.0], [-0.34, 0.0], [0.0, 0.34], [0.0, -0.34]]

## Generator and verifier illustration

# Generator:
np = 10
α_list = (0:np-1)*2π/np
xs = map(α -> [cos(α), sin(α)], α_list)
const WT_ = CPL.Witness{Vector{Float64},Float64,Vector{Float64}}
wit_cls = Vector{WT_}[]
for x in xs
    local wit_cl = WT_[]
    for piece in pieces
        any(lf -> dot(lf, x) > eps(), piece.lfs_dom) && continue
        local α = norm(x, 1)*(1 + opnorm(I + τ*piece.A, 1))
        local y = x + τ*(piece.A*x)
        push!(wit_cl, CPL.Witness(x, α, y))
    end
    push!(wit_cls, wit_cl)
end
rmax = 100
lfs, r = CPL.compute_lfs(wit_cls, lfs_init, 2, rmax, solver)

# Verifier:
xmax = 1e4
x, γ, q, flag = CPL.verify(pieces, lfs, 2, xmax, solver)
@assert flag

# Illustration
fig = figure(0, figsize=(8, 10))
ax = fig.add_subplot(aspect="equal")

β = 1.25
αd = 1.25
xlims = (-3, 3)
ylims = (-3, 3)
ax.set_xlim(xlims...)
ax.set_ylim(ylims...)
ax.set_xticks(-3:1:3)
ax.set_yticks(-3:1:3)
ax.tick_params(axis="both", labelsize=15)
ax.plot(xlims, (0, 0), ls="--", c="black", lw=1.0)
ax.plot(0, 0, marker="x", ms=10, c="black", mew=2.5)

for piece in pieces
    plot_field!(ax, piece, xlims, ylims, 20)
end
lims = ((-10, -10), (10, 10))
plot_level2D!(ax, lfs, β, lims, fc="gold", ec="gold", ew=2)
plot_wits!(ax, wit_cls, lfs, β, αd)

α = norm(x, 1)*(1 + opnorm(I + τ*pieces[q].A, 1))
y = x + τ*(pieces[q].A*x)
plot_wits!(ax, [[CPL.Witness(x, α, y)]], lfs, β, αd, mc="k", lc="r")

ax.text(1.5, +1.6, L"q=1",
        horizontalalignment="center", verticalalignment="center",
        fontsize=20, alpha=1.0, bbox=Dict(["facecolor"=>"white", "alpha"=>1.0]))
ax.text(1.5, -1.6, L"q=2",
        horizontalalignment="center", verticalalignment="center",
        fontsize=20, alpha=1.0, bbox=Dict(["facecolor"=>"white", "alpha"=>1.0]))

LH = (matplotlib.patches.Patch(fc="gold", ec="gold", lw=2.5, alpha=0.5,
        label=L"V(x)\leq1"),)
ax.legend(handles=LH, fontsize=20, loc="upper left",
          facecolor="white", framealpha=1.0)

fig.savefig(string(
    @__DIR__, "/../figures/fig_exa_illustrative_generator_verifier.png"
), dpi=200, transparent=false, bbox_inches="tight")

## Learner feasible illustration
tol_r = 5e-3 # 0.1
xmax = 1e4
iter_max = 50

const wit_cls_list = Vector{Vector{WT_}}[]
const lfs_list = Vector{Vector{Float64}}[]
const x_list = Vector{Float64}[]
const q_list = Int[]

function callback_fcn(::Any, wit_cls, lfs, x, q)
    push!(wit_cls_list, copy(wit_cls))
    push!(lfs_list, copy(lfs))
    push!(x_list, copy(x))
    push!(q_list, q)
end

status, lfs = CPL.learn_lyapunov(
    pieces, lfs_init, τ, 2, xmax, iter_max, solver,
    tol_r=tol_r, callback_fcn=callback_fcn
)

@assert status == CPL.LYAPUNOV_FOUND

fig = figure(2, figsize=(8, 8))
ax_ = fig.subplots(
    nrows=3, ncols=3,
    gridspec_kw=Dict("wspace"=>0.1, "hspace"=>0.1),
    subplot_kw=Dict("aspect"=>"equal")
)
niter = length(lfs_list)
iters = vcat(collect(1:length(ax_)-1), [niter])
nplot = length(iters)

xlims = (-4, 4)
ylims = (-4, 4)

for ax in ax_
    ax.set_xlim(xlims...)
    ax.set_ylim(ylims...)
    ax.set_xticks(())
    ax.set_yticks(())
end

for ax in ax_[:, 1]
    ax.set_yticks(-4:2:4)
end

for ax in ax_[size(ax_)[1], :]
    ax.set_xticks(-4:2:4)
end

lims = ((-10, -10), (10, 10))
β = 1
αd = 1

for k = 1:nplot
    ax = ax_[k]
    ax.plot(xlims, (0, 0), ls="--", c="black", lw=0.5)
    ax.plot(0, 0, marker="x", ms=5, c="black", mew=1.5)
    local iter = iters[k]
    local lfs = lfs_list[iter]
    plot_level2D!(ax, lfs, β, lims, fc="gold", ec="gold", ew=1)
    local wit_cls = wit_cls_list[iter]
    plot_wits!(ax, wit_cls, lfs, β, αd, ms=7.5, lw=1.5)
    if iter < niter
        local x = x_list[iter]
        local q = q_list[iter]
        local α = norm(x, 1)*(1 + opnorm(I + τ*pieces[q].A, 1))
        local y = x + τ*(pieces[q].A*x)
        plot_wits!(
            ax, [[CPL.Witness(x, α, y)]], lfs,
            β, αd, mc="k", lc="r", ms=7.5, lw=1.5
        )
    end
    ax.text(
        0.0, 3.75, string("Step ", iter),
        horizontalalignment="center", fontsize=14, backgroundcolor="w"
    )
end

fig.savefig(string(
    @__DIR__, "/../figures/fig_exa_illustrative_learner_steps.png"
), dpi=200, transparent=false, bbox_inches="tight")

## Learner infeasible
tol_r = 1e-2 # 0.1
xmax = 1e4
iter_max = 50

status, lfs = CPL.learn_lyapunov(
    pieces, lfs_init, τ, 2, xmax, iter_max, solver, tol_r=tol_r
)

@assert status == CPL.LYAPUNOV_INFEASIBLE

end # module