module ExampleIllustrative

using LinearAlgebra
using JuMP
using Gurobi
using PyPlot

include("../../../src/CEGISPolyhedralLyapunov.jl")
CPL = CEGISPolyhedralLyapunov
CPLA = CPL.AdaptiveComplexity
CPLP = CPL.Polyhedra
include("../../utils/geometry.jl")

const GUROBI_ENV = Gurobi.Env()
solver = optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
)

## Parameters
ϵ = 1e1
θ = 0.1
δ = 1e-6
nvar = 2
prob = CPLA.LearningProblem(nvar, ϵ, θ, δ)
CPLA.set_tol_rad!(prob, 1e-3)

α = 1.5
CPLA.set_Gs!(prob, α)

domain = CPLP.Cone()
CPLP.add_supp!(domain, CPLP.Supp([0.0, -1.0]))
A = [-0.5 1.0; -1.0 -0.5]
CPLA.add_system!(prob, domain, A)

domain = CPLP.Cone()
CPLP.add_supp!(domain, CPLP.Supp([0.0, 1.0]))
A = [0.01 1.0; -1.0 0.01]
CPLA.add_system!(prob, domain, A)

## Generator illustration
vecsgen = CPLA.VecsGenerator(nvar, ϵ, θ, δ, prob.Gs)

np = 10
α_list = range(0, 2π, length=np + 1)[1:np]
points = map(α -> [cos(α), sin(α)], α_list)

witnesses = CPLA.Witness[]
for point in points
    wit = CPLA._add_vecs_point!(vecsgen, prob.systems, point)
    push!(witnesses, wit)
end

vecs, r = CPLA.compute_vecs(vecsgen, solver)

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
    CPLP.add_halfspace!(p, CPLP.Halfspace(vec, -1.0))
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
    point_norm = maximum(vec -> dot(vec, wit.point), vecs)
    for deriv in wit.derivs
        derivs_norm = max(derivs_norm, norm(deriv)/point_norm)
    end
end

deriv_length = 0.6

for wit in witnesses
    point_norm = maximum(vec -> dot(vec, wit.point), vecs)
    point_scaled = wit.point*scaling/point_norm
    ax.plot(point_scaled..., marker=".", ms=15, c="blue")
    for deriv in wit.derivs
        deriv_scaled = deriv/(point_norm*derivs_norm)
        point2_scaled = point_scaled + deriv_length*deriv_scaled
        ax.plot(
            (point_scaled[1], point2_scaled[1]),
            (point_scaled[2], point2_scaled[2]),
            c="green", lw=2.5
        )
    end
end

ax.text(1.5, +1.6, L"\mathcal{Q}(x)=1",
        horizontalalignment="center", verticalalignment="center", fontsize=20)
ax.text(1.5, -1.6, L"\mathcal{Q}(x)=2",
        horizontalalignment="center", verticalalignment="center", fontsize=20)

LH = (matplotlib.patches.Patch(fc="gold", ec="gold", lw=2.5, alpha=0.5,
        label=L"V(x)\leq1"),)
ax.legend(handles=LH, fontsize=20, loc="upper left")

fig.savefig(
    "./examples/figures/AdaptiveComplexity/fig_exa_illustrative_generator.png",
    dpi=200, transparent=false, bbox_inches="tight"
)

## Verifier illustration
verifs_lie = CPLA._make_verifs(prob.nvar, prob.systems)[2]

np = 10
α_list = range(0, 2π, length=np + 1)[1:np]
vecs = map(α -> [cos(α), sin(α)], α_list)

x, val, q = CPL.verify(verifs_lie, vecs, solver)
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
    CPLP.add_halfspace!(p, CPLP.Halfspace(vec, -1.0))
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

for system in prob.systems
    D = Matrix{Vector{Float64}}(undef, ngrid, ngrid)
    for (k, x) in enumerate(X)
        if x ∈ system.domain
            D[k] = system.A*x
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
deriv = prob.systems[q].A*point
deriv_scaled = deriv/(point_norm*norm(deriv))
point2_scaled = point_scaled + deriv_length*deriv_scaled
ax.plot(
    (point_scaled[1], point2_scaled[1]),
    (point_scaled[2], point2_scaled[2]),
    c="green", lw=2.5
)

ax.text(1.5, +1.6, L"\mathcal{Q}(x)=1",
        horizontalalignment="center", verticalalignment="center",
        fontsize=20, alpha=1.0, bbox=Dict(["facecolor"=>"white", "alpha"=>1.0]))
ax.text(1.5, -1.6, L"\mathcal{Q}(x)=2",
        horizontalalignment="center", verticalalignment="center",
        fontsize=20, alpha=1.0, bbox=Dict(["facecolor"=>"white", "alpha"=>1.0]))

LH = (matplotlib.patches.Patch(fc="gold", ec="gold", lw=2.5, alpha=0.5,
        label=L"V(x)\leq1"),)
ax.legend(handles=LH, fontsize=20, loc="upper left",
          facecolor="white", framealpha=1.0)

fig.savefig(
    "./examples/figures/AdaptiveComplexity/fig_exa_illustrative_verifier.png",
    dpi=200, transparent=false, bbox_inches="tight")

## Learner illustration
points_init = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
for point in points_init
    CPLA.add_point_init!(prob, point)
end

flag = CPLA.learn_lyapunov!(prob, 100, solver)

@assert flag

fig = figure(2, figsize=(8, 8))
ax_ = fig.subplots(
    nrows=3, ncols=3,
    gridspec_kw=Dict("wspace"=>0.1, "hspace"=>0.1),
    subplot_kw=Dict("aspect"=>"equal")
)
nvecs = length(prob.vecs_list)
indexes = unique(round.(Int, range(1, nvecs, length=length(ax_))))
nplot = length(indexes)

xlims = (-2, 2)
ylims = (-2, 2)

for (k, ax) in enumerate(ax_)
    ax.set_xlim(xlims...)
    ax.set_ylim(ylims...)
    ax.set_xticks(())
    ax.set_yticks(())
    if k in (1, 2, 3)
        ax.set_yticks(-2:1:2)
    end
    if k in (3, 6, 9)
        ax.set_xticks(-2:1:2)
    end
end

verts_radius = -Inf
verts_list = Vector{Vector{Vector{Float64}}}(undef, nplot)

for k = 1:nplot
    global verts_radius
    idx = indexes[k]
    vecs = prob.vecs_list[idx]
    local p = CPLP.Polyhedron()
    for vec in vecs
        CPLP.add_halfspace!(p, CPLP.Halfspace(vec, -1.0))
    end
    verts = compute_vertices_2d(p)
    verts_list[k] = verts
    verts_radius = max(
        verts_radius, maximum(vert -> norm(vert, Inf), verts)
    )
end

derivs_norm = -Inf

for idx = 1:min(maximum(indexes) + 1, nvecs)
    global derivs_norm
    for wit in prob.witnesses_list[idx]
        point_norm = maximum(vec -> dot(vec, wit.point), prob.vecs_list[idx])
        for deriv in wit.derivs
            derivs_norm = max(derivs_norm, norm(deriv)/point_norm)
        end
    end
end

scaling = 1.8/verts_radius
deriv_length = 0.6
kplot_map = [1, 4, 7, 2, 5, 8, 3, 6, 9]

for k = 1:nplot
    ax = ax_[kplot_map[k]]
    idx = indexes[k]
    ax.plot(xlims, (0, 0), ls="--", c="black", lw=0.5)
    ax.plot(0, 0, marker="x", ms=5, c="black", mew=1.5)
    vecs = prob.vecs_list[idx]
    verts_scaled = map(vert -> vert*scaling, verts_list[k])
    polylist = matplotlib.collections.PolyCollection([verts_scaled])
    fca = matplotlib.colors.colorConverter.to_rgba("gold", alpha=0.5)
    polylist.set_facecolor(fca)
    polylist.set_edgecolor("gold")
    polylist.set_linewidth(1.0)
    ax.add_collection(polylist)
    for idx_sub in 1:idx
        for wit in prob.witnesses_list[idx_sub]
            point_norm = maximum(vec -> dot(vec, wit.point), vecs)
            point_scaled = wit.point*scaling/point_norm
            ax.plot(point_scaled..., marker=".", ms=7.5, c="blue")
            for deriv in wit.derivs
                deriv_scaled = deriv/(point_norm*derivs_norm)
                point2_scaled = point_scaled + deriv_length*deriv_scaled
                ax.plot(
                    (point_scaled[1], point2_scaled[1]),
                    (point_scaled[2], point2_scaled[2]),
                    c="green", lw=1.5
                )
            end
        end
    end
    if idx < nvecs
        for wit in prob.witnesses_list[idx + 1]
            point_norm = maximum(vec -> dot(vec, wit.point), vecs)
            point_scaled = wit.point*scaling/point_norm
            ax.plot(point_scaled..., marker=".", ms=7.5, c="black")
            for deriv in wit.derivs
                deriv_scaled = deriv/(point_norm*derivs_norm)
                point2_scaled = point_scaled + deriv_length*deriv_scaled
                ax.plot(
                    (point_scaled[1], point2_scaled[1]),
                    (point_scaled[2], point2_scaled[2]),
                    c="red", lw=1.5
                )
            end
        end
    end
    ax.text(
        0.0, 1.6, string("Step ", idx),
        horizontalalignment="center", fontsize=14
    )
end

x0 = [1.0, -1e-6]
idx = indexes[nplot]
x = x0*scaling/maximum(vec -> dot(vec, x0), prob.vecs_list[idx])
ax = ax_[nplot]

ax.plot(x..., marker=".", ms=7.5, c="purple")

nstep = 100
dt = 4π/nstep
xplot_seq = [x]

for t = 1:nstep-1
    global x
    q = 0
    for system in prob.systems
        if x ∈ system.domain
            x = exp(system.A*dt)*x
            break
        end
    end
    push!(xplot_seq, x)
end

ax.plot(getindex.(xplot_seq, 1), getindex.(xplot_seq, 2), lw=1.5, c="purple")

fig.savefig(
    "./examples/figures/AdaptiveComplexity/fig_exa_illustrative_learner.png",
    dpi=200, transparent=false, bbox_inches="tight"
)

end # module