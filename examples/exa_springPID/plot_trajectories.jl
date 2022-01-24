module TestMain

using LinearAlgebra
using PyPlot
using DifferentialEquations

datafile = "data_set_3"
include(string("./", datafile, ".jl"))

function MassSpringPID!(du, u, p, t)
    # u = [∫y, x, x']
    Ki, Kp, Kd = p
    if Kd*u[3] + Kp*u[2] + Ki*u[1] ≥ 0
        du[1] = u[2]
        du[2] = u[3]
        du[3] = - Ki*u[1] - (a0 + Kp)*u[2] - (a1 + Kd)*u[3]
    else
        du[1] = - c_aw*u[1] + u[2]
        du[2] = u[3]
        du[3] = - a0*u[2] - a1*u[3]
    end
end

p0 = (0, 0, 0)
p1 = (Ki, Kp, Kd)
p_list = (p0, p1)
u0p = [0.0, 1.0, 0.0]
u0n = [0.0, -1.0, 0.0]
u0s = [-1.0, 0.0, 0.0]
u0_list = (u0p, u0n)

fig = figure(figsize=(12, 10))
ax_ = fig.subplots(nrows=4, gridspec_kw=Dict("hspace"=>0.1))

for p in p_list, u0 in u0_list
    tspan = (0.0, 10.0)
    prob = ODEProblem(MassSpringPID!, u0, tspan, p)
    sol = solve(prob)
    for i = 1:3
        ax_[i].plot(sol.t, sol[i, :])
    end
    ax_[4] = plot(sol.t, p[1]*sol[1, :] + p[2]*sol[2, :] + p[3]*sol[3, :])
end

end # TestMain