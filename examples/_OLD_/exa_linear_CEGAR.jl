module TestMain

using LinearAlgebra
using Random
using JuMP
using Gurobi
using PyPlot

Random.seed!(10)
solver = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => false)

## Parameters
A = [-0.1 1.0; -1.0 -0.1]
A = [-0.2 2.0; -0.5 -0.2]
# A = [-0.2 1.0 0.0; -1.0 -0.2 0.0; 0.0 0.0 -1.0]
N = 5
D = 2
x_list = [randn(D) for i = 1:N]
dx_list = map(x -> A*x, x_list)

## Solving
while true
    # Learning Phase
    display("New Learning")
    # readline()
    global N = length(x_list)
    model = Model(solver)
    c_list = [@variable(model, [1:D], base_name=string("c", i),
        lower_bound=-1.0, upper_bound=1.0) for i = 1:N]
    r = @variable(model)
    M = 3

    for i = 1:N
        xt = x_list[i]
        dxt = dx_list[i]
        nxt = norm(xt)
        x = xt/nxt
        dx = dxt/nxt
        ndx = norm(dx)
        c = c_list[i]
        @constraint(model, dx'*c + ndx*r ≤ 0)
        for j = 1:N
            j == i && continue
            d = c_list[j]
            @constraint(model, x'*(c + d) ≥ 0)
            @constraint(model, x'*(c - d) ≥ 0)
            @constraint(model, (+dx)'*d - M*x'*(c - d) + ndx*r ≤ 0)
            @constraint(model, (-dx)'*d - M*x'*(c + d) + ndx*r ≤ 0)
        end
    end

    @objective(model, Max, r)

    optimize!(model)
    display(termination_status(model))
    display(primal_status(model))
    display(dual_status(model))

    global copt_list = map(c -> value.(c), c_list)

    # Verification Phase
    x_counter = Vector{Float64}[]

    for i = 1:N
        c = copt_list[i]
        model = Model(solver)
        x = @variable(model, [1:D], lower_bound=-1.0, upper_bound=1.0)

        for j = 1:N
            j == i && continue
            d = copt_list[j]
            @constraint(model, (c + d)'*x ≥ 0)
            @constraint(model, (c - d)'*x ≥ 0)
        end

        @objective(model, Max, c'*(A*x))

        optimize!(model)
        display(termination_status(model))
        display(primal_status(model))
        display(dual_status(model))
        display(objective_value(model))
        
        if objective_value(model) ≥ 0.001
            push!(x_counter, value.(x))
            display(x_counter)
        end
    end

    isempty(x_counter) && break
    append!(x_list, x_counter)
    append!(dx_list, map(x -> A*x, x_counter))
end

## Plotting
fig = figure(0, figsize=(12, 10))
ax = fig.add_subplot(aspect="equal")
xlims = (-2, 2)
ylims = (-2, 2)
ax.set_xlim(xlims...)
ax.set_ylim(ylims...)

np = 600
xf1 = range(xlims[1], stop = xlims[2], length = np)
xf2 = range(ylims[1], stop = ylims[2], length = np)
Xftemp = collect(Iterators.product(xf1, xf2))
Xf = map(x -> [x...], Xftemp)
X1 = map(x -> x[1], Xf)
X2 = map(x -> x[2], Xf)

norm_poly(x) = maximum(c -> abs(c'*x), copt_list)
Z = map(x -> norm_poly(x), Xf)
zm = min(minimum(Z[1, :]), minimum(Z[:, 1]))/1.2

ax.contour(X1, X2, Z, levels=(-1, zm), colors="yellow")
h = ax.contourf(X1, X2, Z, levels=(-1, zm), colors = "none", hatches="//")
for coll in h.collections
    coll.set_edgecolor("yellow")
end

α = 0.2
xdxs_list = map((x, dx) -> (x*zm/norm_poly(x), dx*zm/norm_poly(x)), x_list, dx_list)

for xdxs in xdxs_list
    ax.plot(xdxs[1]..., marker=".", ms=10, c="red")
    arr = (xdxs[1], xdxs[1] + α*xdxs[2])
    ax.plot((arr[1][1], arr[2][1]), (arr[1][2], arr[2][2]), c="green")
end

end # TestMain