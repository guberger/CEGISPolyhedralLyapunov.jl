module ExampleHybrid

# In work!!!
# For the moment: only plotting ...

using LinearAlgebra
using JuMP
using Gurobi
using PyPlot

include("../../src/CEGISPolyhedralLyapunov.jl")
CPL = CEGISPolyhedralLyapunov
Cone = CPL.Cone
System = CPL.System

include("../utils/geometry.jl")
include("plotting.jl")

const GUROBI_ENV = Gurobi.Env()
solver() = Model(optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
))

## Parameters
τ = 0.0
ϵ = 10.0
δ = 0.1
nvar = 2
nloc = 2

sys = System()

θ = π/7.5
R = [cos(θ) -sin(θ); sin(θ) cos(θ)]

domain = Cone()
CPL.add_supp!(domain, R \ [0.0, -1.0])
α = 0.9
CPL.add_piece_disc!(sys, domain, 1, α*R, 1)

domain = Cone()
CPL.add_supp!(domain, R \ [0.0, 1.0])
α = 0.9
CPL.add_piece_disc!(sys, domain, 1, α*R, 2)

θ = π/5.5
R = [cos(θ) -sin(θ); sin(θ) cos(θ)]

domain = Cone()
CPL.add_supp!(domain, R \ [0.0, -1.0])
α = 1.3
CPL.add_piece_disc!(sys, domain, 2, α*R, 1)

domain = Cone()
CPL.add_supp!(domain, R \ [0.0, 1.0])
α = 1.1
CPL.add_piece_disc!(sys, domain, 2, α*R, 2)

fig = figure(0, figsize=(8, 10))
ax_ = fig.subplots(
    nrows=1, ncols=2,
    gridspec_kw=Dict("wspace"=>0.1, "hspace"=>0.1),
    subplot_kw=Dict("aspect"=>"equal")
)

for ax in ax_
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
end

x0 = [1.0, -1e-6]
loc0 = 1
nstep = 9
dt = 4π/nstep

plot_traj!(ax_, sys, x0, loc0, dt, nstep)

end # module