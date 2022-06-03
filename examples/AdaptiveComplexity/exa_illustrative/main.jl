module ExampleIllustrative

using LinearAlgebra
using JuMP
using Gurobi
using PyPlot

include("../../../src/CEGISPolyhedralLyapunov.jl")
CPLA = CEGISPolyhedralLyapunov.AdaptiveComplexity
CPLP = CEGISPolyhedralLyapunov.Polyhedra
include("../../utils/geometry.jl")
include("plotting.jl")

const GUROBI_ENV = Gurobi.Env()
solver = optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
)

## Parameters
ϵ = 10.0
θ = 1/3.2 # 0.1
δ = 0.025 # 0.1
nvar = 2

sys = CPLA.System()

domain = CPLP.Cone()
CPLP.add_supp!(domain, [0.0, -1.0])
A = [-0.5 1.0; -1.0 -0.5]
CPLA.add_piece!(sys, domain, A)

domain = CPLP.Cone()
CPLP.add_supp!(domain, [0.0, 1.0])
A = [0.01 1.0; -1.0 0.01]
CPLA.add_piece!(sys, domain, A)

## Generator and verifier illustration

# Generator:
gen = CPLA.Generator(nvar)
np = 10
α_list = range(0, 2π, length=np + 1)[1:np]
points = map(α -> [cos(α), sin(α)], α_list)
witnesses = CPLA.Witness[]
for point in points
    wit = CPLA.make_witness_from_point_system(sys, point)
    CPLA.add_witness!(gen, wit)
    push!(witnesses, wit)
end
# vecs, r = CPLA.compute_vecs_chebyshev(gen, 1/θ, solver)
vecs, r = CPLA.compute_vecs_witness(gen, 1/θ, solver) # test

# Verifier:
verif = CPLA.make_verif_from_system(nvar, sys)
x, val, q = CPLA.verify_lie(verif, vecs, solver)
ce_point = x/norm(x, Inf)

# Illustration
fig = figure(0, figsize=(8, 10))
ax = fig.add_subplot(aspect="equal")

xlims = (-2, 2)
ylims = (-2, 2)
ax.set_xlim(xlims...)
ax.set_ylim(ylims...)
ax.set_xticks(-2:1:2)
ax.set_yticks(-2:1:2)
ax.tick_params(axis="both", labelsize=15)
ax.plot(xlims, (0, 0), ls="--", c="black", lw=1.0)
ax.plot(0, 0, marker="x", ms=10, c="black", mew=2.5)

plot_field!(ax, sys, xlims, ylims, 20)
level = plot_level!(ax, vecs, 1.8)
plot_witnesses!(ax, witnesses, vecs, level, 0.6)

ce_wit = CPLA.make_witness_from_point_system(sys, ce_point)
plot_witnesses!(ax, (ce_wit,), vecs, level, 0.6, mc="black", lc="red")

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
        @__DIR__,
        "/../../figures/AdaptiveComplexity/",
        "fig_exa_illustrative_generator_verifier.png"
    ),
    dpi=200, transparent=false, bbox_inches="tight")

## Learner feasible illustration
lear = CPLA.Learner(nvar, sys, ϵ, θ, δ)
CPLA.set_tol!(lear, :rad, 1e-3)

points_init = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
for point in points_init
    CPLA.add_point_init!(lear, point)
end

sol = CPLA.learn_lyapunov!(lear, 100, solver)

@assert sol.status == CPLA.LYAPUNOV_FOUND

fig = figure(2, figsize=(8, 8))
ax_ = fig.subplots(
    nrows=3, ncols=3,
    gridspec_kw=Dict("wspace"=>0.1, "hspace"=>0.1),
    subplot_kw=Dict("aspect"=>"equal")
)
nvecs = length(sol.vecs_list)
indexes = unique(round.(Int, range(1, nvecs, length=length(ax_))))
nplot = length(indexes)

xlims = (-2, 2)
ylims = (-2, 2)

for ax in ax_
    ax.set_xlim(xlims...)
    ax.set_ylim(ylims...)
    ax.set_xticks(())
    ax.set_yticks(())
end

for ax in ax_[:, 1]
    ax.set_yticks(-2:1:2)
end

for ax in ax_[size(ax_)[1], :]
    ax.set_xticks(-2:1:2)
end

radmax = 1.7
deriv_length = 0.6

for k = 1:nplot
    ax = ax_[LinearIndices(ax_)'[k]]
    ax.plot(xlims, (0, 0), ls="--", c="black", lw=0.5)
    ax.plot(0, 0, marker="x", ms=5, c="black", mew=1.5)
    idx = indexes[k]
    vecs = sol.vecs_list[idx]
    level = plot_level!(ax, vecs, radmax, ew=1)
    witnesses = sol.witnesses_list[idx]
    plot_witnesses!(
        ax, witnesses, vecs, level, deriv_length, ms=7.5, lw=1.5
    )
    if idx ≤ length(sol.counterexample_list)
        ce_wit = sol.counterexample_list[idx]
        plot_witnesses!(
            ax, (ce_wit,), vecs, level, deriv_length,
            mc="black", ms=7.5, lc="red", lw=1.5
        )
    else
        # plot trajectory on last plot:
        x0 = [1.0, -1e-6]
        x0_scaled = x0*level/_norm(vecs, x0)
        nstep = 100
        dt = 4π/nstep
        plot_traj!(ax, sys, x0_scaled, dt, nstep, ms=7.5, lw=1.5)
    end
    ax.text(
        0.0, 1.6, string("Step ", idx),
        horizontalalignment="center", fontsize=14
    )
end

fig.savefig(string(
        @__DIR__,
        "/../../figures/AdaptiveComplexity/",
        "fig_exa_illustrative_learner_steps.png"
    ),
    dpi=200, transparent=false, bbox_inches="tight")

## Learner infeasible illustration
δ = 0.05

lear = CPLA.Learner(nvar, sys, ϵ, θ, δ)
CPLA.set_tol!(lear, :rad, 1e-3)

points_init = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
for point in points_init
    CPLA.add_point_init!(lear, point)
end

sol = CPLA.learn_lyapunov!(lear, 100, solver)
display(sol.status)
display(sol.niter)

end # module