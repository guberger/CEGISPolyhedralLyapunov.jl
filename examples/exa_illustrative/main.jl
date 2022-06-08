module ExampleIllustrative

using LinearAlgebra
using JuMP
using Gurobi
using PyPlot

include("../../src/CEGISPolyhedralVerification.jl")
CPV = CEGISPolyhedralVerification
Cone = CPV.Cone
System = CPV.System
Witness = CPV.Witness

include("../utils/geometry.jl")
include("plotting.jl")

const GUROBI_ENV = Gurobi.Env()
solver = optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
)

# With τ = 1/4, ϵ = 10, δ = 0.05; with nD = opnorm(A - I)
# finds PLF in 30 steps
# With δ = 0.1, infeasible in 12 steps

## Parameters
τ = 1/4 # 0.1
ϵ = 10.0
δ = 0.05 # 0.1
nvar = 2
nloc = 1

sys = System()

domain = Cone()
CPV.add_supp!(domain, [0.0, -1.0])
A = [-0.5 1.0; -1.0 -0.5]
CPV.add_piece_cont!(sys, domain, 1, A)

domain = Cone()
CPV.add_supp!(domain, [0.0, 1.0])
A = [0.01 1.0; -1.0 0.01]
CPV.add_piece_cont!(sys, domain, 1, A)

## Generator and verifier illustration

# Generator:
gen = CPV.Generator(nvar, nloc)

np = 10
α_list = range(0, 2π, length=np + 1)[1:np]
points = map(α -> [cos(α), sin(α)], α_list)
for point in points
    wit = Witness(1, point/norm(point, Inf))
    CPV._add_evidences!(gen, sys, τ, wit)
end
# mpf, r = CPV.compute_mpf_chebyshev(gen, solver)
mpf, r = CPV.compute_mpf_evidence(gen, solver) # test

# Verifier:
verif = CPV.Verifier()
CPV._add_predicates!(verif, nvar, sys)
x, r_liecont, loc = CPV.verify_lie_cont(verif, mpf, solver)

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

for piece in sys.cont_pieces
    plot_field!(ax, piece, xlims, ylims, 20)
end
level = plot_level!(ax, mpf.pfs[1], 1.8)
plot_evids!(ax, gen.pos_evids, mpf, level)
plot_evids!(ax, gen.liecont_evids, mpf, level, 0.6)

old_pos_evids = copy(gen.pos_evids)
old_liecont_evids = copy(gen.liecont_evids)

CPV._add_evidences!(gen, sys, τ, Witness(loc, x/norm(x, Inf)))
new_pos_evids = setdiff(gen.pos_evids, old_pos_evids)
new_liecont_evids = setdiff(gen.liecont_evids, old_liecont_evids)
plot_evids!(ax, new_pos_evids, mpf, level, mc="black")
plot_evids!(ax, new_liecont_evids, mpf, level, 0.6, mc="black", lc="red")

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
lear = CPV.Learner(nvar, loc, sys, τ, ϵ, δ)
CPV.set_tol!(lear, :rad, 1e-3)

points_init = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
for point in points_init
    CPV.add_witness!(lear, 1, point)
end

tracerec = CPV.TraceRecorder()
status = CPV.learn_lyapunov!(lear, 100, solver, tracerec=tracerec)[1]

@assert status == CPV.LYAPUNOV_FOUND

fig = figure(2, figsize=(8, 8))
ax_ = fig.subplots(
    nrows=3, ncols=3,
    gridspec_kw=Dict("wspace"=>0.1, "hspace"=>0.1),
    subplot_kw=Dict("aspect"=>"equal")
)
nmpf = length(tracerec.mpf_list)
indexes = unique(round.(Int, range(1, nmpf, length=length(ax_))))
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
    mpf = tracerec.mpf_list[idx]
    level = plot_level!(ax, mpf.pfs[1], radmax, ew=1)
    pos_evids = tracerec.pos_evids_list[idx]
    plot_evids!(ax, pos_evids, mpf, level, ms=7.5)
    liecont_evids = tracerec.liecont_evids_list[idx]
    plot_evids!(
        ax, liecont_evids, mpf, level, deriv_length, ms=7.5, lw=1.5
    )
    if idx < nmpf
        new_pos_evids = setdiff(
            tracerec.pos_evids_list[idx + 1], tracerec.pos_evids_list[idx]
        )
        plot_evids!(ax, new_pos_evids, mpf, level, mc="black", ms=7.5)
        new_liecont_evids = setdiff(
            tracerec.liecont_evids_list[idx + 1],
            tracerec.liecont_evids_list[idx]
        )
        plot_evids!(
            ax, new_liecont_evids, mpf, level, deriv_length,
            mc="black", ms=7.5, lc="red", lw=1.5
        )
    else
        # plot trajectory on last plot:
        x0 = [1.0, -1e-6]
        x0_scaled = x0*level/_norm(mpf.pfs[1], x0)
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
    @__DIR__, "/../figures/fig_exa_illustrative_learner_steps.png"
), dpi=200, transparent=false, bbox_inches="tight")

## Learner infeasible illustration
δ = 0.1
lear = CPV.Learner(nvar, nloc, sys, τ, ϵ, δ)
CPV.set_tol!(lear, :rad, 1e-3)

points_init = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
for point in points_init
    CPV.add_witness!(lear, 1, point)
end

status, mpf, niter = CPV.learn_lyapunov!(lear, 100, solver)
display(status)
display(niter)

end # module