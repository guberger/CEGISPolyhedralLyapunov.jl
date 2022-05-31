module ExampleIllustrative

using LinearAlgebra
using JuMP
using Gurobi
using PyPlot

include("../../../src/CEGISPolyhedralLyapunov.jl")
CPLA = CEGISPolyhedralLyapunov.AdaptiveComplexity
CPLP = CEGISPolyhedralLyapunov.Polyhedra
include("../../utils/geometry.jl")

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

## Generator illustration
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

vecs, r = CPLA.compute_vecs_chebyshev(gen, 1/θ, solver)

fig = figure(0, figsize=(8, 10))
ax = fig.add_subplot(aspect="equal")

xlims = (-2, 2)
ylims = (-2, 2)
ax.set_xlim(xlims...)
ax.set_ylim(ylims...)
ax.set_xticks(-2:1:2)
ax.set_yticks(-2:1:2)
ax.tick_params(axis="both", labelsize=15)

p = CPLP.Polyhedron()
for vec in vecs
    CPLP.add_halfspace!(p, vec, -1.0)
end
verts = compute_vertices_2d(p)
verts_radius = maximum(vert -> norm(vert, Inf), verts)
scaling = 1.8/verts_radius
verts_scaled = map(vert -> vert*scaling, verts)

ax.plot(xlims, (0, 0), ls="--", c="black", lw=1.0)
ax.plot(0, 0, marker="x", ms=10, c="black", mew=2.5)
polylist = matplotlib.collections.PolyCollection([verts_scaled])
fca = matplotlib.colors.colorConverter.to_rgba("gold", alpha=0.5)
polylist.set_facecolor(fca)
polylist.set_edgecolor("gold")
polylist.set_linewidth(2.0)
ax.add_collection(polylist)

derivs_norm = -Inf

for wit in witnesses
    global derivs_norm
    for lieevid in wit.lie_evids
        point_norm = maximum(vec -> dot(vec, lieevid.point), vecs)
        derivs_norm = max(derivs_norm, norm(lieevid.deriv)/point_norm)
    end
end

deriv_length = 0.6

for wit in witnesses
    for posevid in wit.pos_evids
        point_norm = maximum(vec -> dot(vec, posevid.point), vecs)
        point_scaled = posevid.point*scaling/point_norm
        ax.plot(point_scaled..., marker=".", ms=15, c="blue")
    end
    for lieevid in wit.lie_evids
        point_norm = maximum(vec -> dot(vec, lieevid.point), vecs)
        point_scaled = lieevid.point*scaling/point_norm
        ax.plot(point_scaled..., marker=".", ms=15, c="blue")
        deriv_scaled = lieevid.deriv/(point_norm*derivs_norm)
        point2_scaled = point_scaled + deriv_length*deriv_scaled
        ax.plot(
            (point_scaled[1], point2_scaled[1]),
            (point_scaled[2], point2_scaled[2]),
            c="green", lw=2.5
        )
    end
end

ax.text(1.5, +1.6, L"q=1",
        horizontalalignment="center", verticalalignment="center", fontsize=20)
ax.text(1.5, -1.6, L"q=2",
        horizontalalignment="center", verticalalignment="center", fontsize=20)

LH = (matplotlib.patches.Patch(fc="gold", ec="gold", lw=2.5, alpha=0.5,
        label=L"V(x)\leq1"),)
ax.legend(handles=LH, fontsize=20, loc="upper left")

fig.savefig(string(
        @__DIR__,
        "/../../figures/AdaptiveComplexity/fig_exa_illustrative_generator.png"
    ),
    dpi=200, transparent=false, bbox_inches="tight")

## Verifier illustration
verif = CPLA.make_verif_from_system(nvar, sys)

np = 10
α_list = range(0, 2π, length=np + 1)[1:np]
vecs = map(α -> [cos(α), sin(α)], α_list)

x, val, q = CPLA.verify_lie(verif, vecs, solver)
point = x/norm(x, Inf)

fig = figure(1, figsize=(8, 10))
ax = fig.add_subplot(aspect="equal")

xlims = (-2, 2)
ylims = (-2, 2)
ax.set_xlim(xlims...)
ax.set_ylim(ylims...)
ax.set_xticks(-2:1:2)
ax.set_yticks(-2:1:2)
ax.tick_params(axis="both", labelsize=15)

p = CPLP.Polyhedron()
for vec in vecs
    CPLP.add_halfspace!(p, vec, -1.0)
end
verts = compute_vertices_2d(p)
verts_radius = maximum(vert -> norm(vert, Inf), verts)
scaling = 1.2/verts_radius
verts_scaled = map(vert -> vert*scaling, verts)

ax.plot(xlims, (0, 0), ls="--", c="black", lw=1.0)
ax.plot(0, 0, marker="x", ms=10, c="black", mew=2.5)
polylist = matplotlib.collections.PolyCollection([verts_scaled])
fca = matplotlib.colors.colorConverter.to_rgba("gold", alpha=0.5)
polylist.set_facecolor(fca)
polylist.set_edgecolor("gold")
polylist.set_linewidth(2.0)
ax.add_collection(polylist)

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

deriv_length = 0.6

point_norm = maximum(vec -> dot(vec, point), vecs)
point_scaled = point*scaling/point_norm
ax.plot(point_scaled..., marker=".", ms=15, c="blue")
deriv = sys.pieces[q].A*point
deriv_scaled = deriv/(point_norm*norm(deriv))
point2_scaled = point_scaled + deriv_length*deriv_scaled
ax.plot(
    (point_scaled[1], point2_scaled[1]),
    (point_scaled[2], point2_scaled[2]),
    c="green", lw=2.5
)

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
        "/../../figures/AdaptiveComplexity/fig_exa_illustrative_verifier.png"
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

levelset_radius = 1.7
deriv_length = 0.6

for k = 1:nplot
    ax = ax_[LinearIndices(ax_)'[k]]
    ax.plot(xlims, (0, 0), ls="--", c="black", lw=0.5)
    ax.plot(0, 0, marker="x", ms=5, c="black", mew=1.5)
    idx = indexes[k]
    vecs = sol.vecs_list[idx]
    # compute verts norm:
    p = CPLP.Polyhedron()
    for vec in vecs
        CPLP.add_halfspace!(p, vec, -1.0)
    end
    verts = compute_vertices_2d(p)
    verts_radius = maximum(vert -> norm(vert, Inf), verts)
    # plot verts:
    scaling = levelset_radius/verts_radius
    verts_scaled = map(vert -> vert*scaling, verts)
    polylist = matplotlib.collections.PolyCollection([verts_scaled])
    fca = matplotlib.colors.colorConverter.to_rgba("gold", alpha=0.5)
    polylist.set_facecolor(fca)
    polylist.set_edgecolor("gold")
    polylist.set_linewidth(1.0)
    ax.add_collection(polylist)
    # compute derivs_norm
    derivs_norm = -Inf
    for wit in sol.witnesses_list[idx]
        for lieevid in wit.lie_evids
            point_norm = maximum(vec -> dot(vec, lieevid.point), vecs)
            derivs_norm = max(derivs_norm, norm(lieevid.deriv)/point_norm)
        end
    end
    # plot derivs
    for wit in sol.witnesses_list[idx]
        for posevid in wit.pos_evids
            point_norm = maximum(vec -> dot(vec, posevid.point), vecs)
            point_scaled = posevid.point*scaling/point_norm
            ax.plot(point_scaled..., marker=".", ms=7.5, c="blue")
        end
        for lieevid in wit.lie_evids
            point_norm = maximum(vec -> dot(vec, lieevid.point), vecs)
            point_scaled = lieevid.point*scaling/point_norm
            ax.plot(point_scaled..., marker=".", ms=7.5, c="blue")
            deriv_scaled = lieevid.deriv/(point_norm*derivs_norm)
            point2_scaled = point_scaled + deriv_length*deriv_scaled
            ax.plot(
                (point_scaled[1], point2_scaled[1]),
                (point_scaled[2], point2_scaled[2]),
                c="green", lw=1.5
            )
        end
    end
    if idx ≤ length(sol.counterexample_list)
        # plot counterexample:
        wit = sol.counterexample_list[idx]
        for posevid in wit.pos_evids
            point_norm = maximum(vec -> dot(vec, posevid.point), vecs)
            point_scaled = posevid.point*scaling/point_norm
            ax.plot(point_scaled..., marker=".", ms=7.5, c="black")
        end
        for lieevid in wit.lie_evids
            point_norm = maximum(vec -> dot(vec, lieevid.point), vecs)
            point_scaled = lieevid.point*scaling/point_norm
            ax.plot(point_scaled..., marker=".", ms=7.5, c="black")
            deriv_scaled = lieevid.deriv/(point_norm*derivs_norm)
            point2_scaled = point_scaled + deriv_length*deriv_scaled
            ax.plot(
                (point_scaled[1], point2_scaled[1]),
                (point_scaled[2], point2_scaled[2]),
                c="red", lw=1.5
            )
        end
    else
        # plot trajectory on last plot:
        x0 = [1.0, -1e-6]
        x = x0*scaling/maximum(vec -> dot(vec, x0), vecs)
        ax.plot(x..., marker=".", ms=7.5, c="purple")
        nstep = 100
        dt = 4π/nstep
        xplot_seq = [x]
        for t = 1:nstep-1
            q = 0
            for system in sys.pieces
                if x ∈ system.domain
                    x = exp(system.A*dt)*x
                    break
                end
            end
            push!(xplot_seq, x)
        end
        ax.plot(
            getindex.(xplot_seq, 1), getindex.(xplot_seq, 2),
            lw=1.5, c="purple"
        )
    end
    ax.text(
        0.0, 1.6, string("Step ", idx),
        horizontalalignment="center", fontsize=14
    )
end

fig.savefig(string(
        @__DIR__,
        "/../../figures/AdaptiveComplexity/fig_exa_illustrative_learner.png"
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