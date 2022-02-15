using LinearAlgebra
using JuMP
using MosekTools
using LazySets
using CDDLib
using PyPlot
using PyCall
const spatial = pyimport_conda("scipy.spatial", "scipy")
MOI = JuMP.MathOptInterface

D = 2
A = [-1 -1; 1 -1; -1 1; 1 1]
b = [0, 1, 1, 3]

poly = HPolytope(Float64.(A), Float64.(b))
points = vertices_list(poly, backend=CDDLib.Library())
verts = convex_hull(points)

fig = figure(0, figsize=(8, 10))
ax = fig.add_subplot(aspect="equal")

xlims = (-1, 2)
ylims = (-1, 2)
ax.set_xlim(xlims...)
ax.set_ylim(ylims...)
ax.set_xticks(-1:1:2)
ax.set_yticks(-1:1:2)
ax.tick_params(axis="both", labelsize=15)

nv = maximum(x -> norm(x, Inf), verts)
scaling = 1.8/nv

verts_scaled = map(x -> x*scaling, verts)
polylist = matplotlib.collections.PolyCollection([verts_scaled])
fca = matplotlib.colors.colorConverter.to_rgba("gold", alpha=0.5)
polylist.set_facecolor(fca)
polylist.set_edgecolor("gold")
polylist.set_linewidth(2.0)
ax.add_collection(polylist)

solver = Mosek.Optimizer

model = Model(solver)

c = @variable(model, [1:D], base_name="c")
B = @variable(model, [1:D, 1:D], base_name="B", PSD)
t = @variable(model, base_name="t")
upper_tri = [B[i, j] for j in 1:D for i in 1:j]

np = 100
α_list = range(0, 2π, length=np)

for con in zip(eachrow(A), b)
    ai, bi = con
    @constraint(model, vcat(bi - dot(ai, c), B*ai) in SecondOrderCone())
end

@constraint(model, vcat(t, upper_tri) in MOI.RootDetConeTriangle(D))

@objective(model, Max, t)

optimize!(model)

println(solution_summary(model))

copt = value.(c)
Bopt = value.(B)

np = 100
α_list = range(0, 2π, length=np)
ellipse = map(α -> Bopt*[cos(α), sin(α)] + copt, α_list)

x1_ell = map(x -> x[1]*scaling, ellipse)
x2_ell = map(x -> x[2]*scaling, ellipse)
ax.plot(x1_ell, x2_ell)

