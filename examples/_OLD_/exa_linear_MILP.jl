module TestMain

using LinearAlgebra
using Random
using JuMP
using Gurobi
using PyPlot

Random.seed!(0)

## Parameters
A = [-0.5 1.0; -1.0 -0.5]
N = 100
x_list = [randn(2) for i = 1:N]
dx_list = map(x -> A*x, x_list)
M = 10

## Solving
solver = optimizer_with_attributes(Gurobi.Optimizer)
model = Model(solver)
c_list = [@variable(model, [1:2], base_name=string("c", i),
    lower_bound=-1.0, upper_bound=1.0) for i = 1:M]
δ_list = [@variable(model, base_name=string("δ", i), binary=true) for i = 1:M]
r = @variable(model)
C_binary = 5000
ϵ_proxy = 0.01

for i = 1:N
    x = x_list[i]/norm(x)
    dx = dx_list[i]/norm(x)
    for j = 1:M
        c = c_list[j]
        for jt = 1:M
    @constraint(model, dx'*c + norm(dx)*r ≤ 0)
    for j = 1:N
        j == i && continue
        d = c_list[j]
        @constraint(model, x'*(+d - c) + norm(x)*r ≤ 0)
        @constraint(model, x'*(-d - c) + norm(x)*r ≤ 0)
    end
end

@objective(model, Max, r)

optimize!(model)
display(solution_summary(model))

copt_list = map(c -> value.(c), c_list)

## Plotting
fig = figure(figsize=(12, 10))
ax = fig.add_subplot(aspect="equal")
xlims = (-2, 2)
ylims = (-2, 2)
ax.set_xlim(xlims...)
ax.set_ylim(ylims...)

α = 0.2
xs_list = map((x, c) -> x/abs(c'*x), x_list, copt_list)
arrow_list = map((x, dx) -> (x, x + α*dx/norm(dx)), xs_list, dx_list)

for (x, arr) in zip(xs_list, arrow_list)
    ax.plot(x..., marker=".", ms=10, c="red")
    arrbis = map(i -> map(e -> e[i], arr), 1:2)
    ax.plot(arrbis..., c="green")
end

np = 600
xf1 = range(xlims[1], stop = xlims[2], length = np)
xf2 = range(ylims[1], stop = ylims[2], length = np)
Xftemp = collect(Iterators.product(xf1, xf2))
Xf = map(x -> [x...], Xftemp)
X1 = map(x -> x[1], Xf)
X2 = map(x -> x[2], Xf)

func_lyapunov(x) = sum(c -> exp(abs(c'*x) - 1) - 1, copt_list)
Z = map(x -> func_lyapunov(x), Xf)
ax.contour(X1, X2, Z, levels=-100:3:0, colors="yellow")
h = ax.contourf(X1, X2, Z, levels=(-1, 1), colors = "none", hatches="//")
for coll in h.collections
    coll.set_edgecolor("yellow")
end

# for i = 1:N
#     c = copt_list[i]
#     Z = map(x -> abs(c'*x), Xf)
    
# end

end # TestMain