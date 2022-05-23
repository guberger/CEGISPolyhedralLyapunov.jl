using LinearAlgebra
using JuMP
const _RSC_ = JuMP.MathOptInterface.ResultStatusCode
const _TSC_ = JuMP.MathOptInterface.TerminationStatusCode

_VT_ = Vector{Float64}

struct Witness
    point::_VT_
    derivs::Vector{_VT_}
end

Witness(point::_VT_) = Witness(point, _VT_[])

mutable struct VecsGenerator
    nvar::Int
    nvec::Int
    witnesses_map::Vector{Tuple{Int,Witness}}
    ϵ::Float64
    Gs::Vector{Float64}
    rs::Vector{Float64}
end

VecsGenerator(nvar::Int, ϵ::Float64, Gs::Vector{Float64}) =
    VecsGenerator(nvar, 0, Tuple{Int,Witness}[], ϵ, Gs, fill(Inf, length(Gs)))

function add_deriv!(wit::Witness, deriv::_VT_)
    push!(wit.derivs, deriv)
end

function add_vec!(vecsgen::VecsGenerator)
    vecsgen.nvec += 1
    return vecsgen.nvec
end

function add_witness!(vecsgen::VecsGenerator, i::Int, wit::Witness)
    push!(vecsgen.witnesses_map, (i, wit))
end

function _add_vars!(model, nvar, nvec)
    vecs = [
        @variable(model, [1:nvar], lower_bound=-1, upper_bound=1) for i = 1:nvec
    ]
    # new:
    avecs = [
        @variable(model, [1:nvar], lower_bound=0, upper_bound=1) for i = 1:nvec
    ]
    for i = 1:nvec
        @constraint(model, -vecs[i] .≤ avecs[i])
        @constraint(model, +vecs[i] .≤ avecs[i])
        @constraint(model, sum(avecs[i]) ≤ 1)
    end # end new
    return vecs
end

function _add_constrs_witness!(model, vecs, r, i, wit, ϵ, G, H)
    point = wit.point
    npoint = norm(wit.point, Inf)
    @constraint(model, dot(point, vecs[i]) ≥ npoint/ϵ) # new
    # @constraint(model, dot(point, vecs[i]) ≥ r*npoint) # not needed
    # @constraint(model, dot(point, vecs[i]) ≥ norm(point)/ϵ) # old
    for deriv in wit.derivs
        nderiv = norm(deriv, Inf) # new test
        nderiv = norm(deriv) # old
        # α = r*npoint # new
        α = r*nderiv # new test
        # α = r*nderiv # old
        @constraint(model, dot(deriv, vecs[i]) + α ≤ 0)
        # β = (H + 1)*r*npoint # new
        β = (H*npoint + nderiv)*r # new test
        # β = r*nderiv # old
        for j = 1:length(vecs)
            j == i && continue
            @constraint(
                model,
                dot(deriv, vecs[j]) - G*dot(point, vecs[i] - vecs[j]) + β ≤ 0
            )
        end
    end
end

# function _add_constr_lie!(model, vec, vec2, wit, β, γ)
#     for deriv in wit.derivs
#         # @constraint(
#         #     model,
#         #     dot(deriv, vec2) -
#         #         γ*dot(wit.point, vec - vec2) +
#         #         β*norm(wit.point, Inf) ≤ 0
#         # ) # new
#         # @constraint(
#         #     model,
#         #     dot(deriv, vec2) -
#         #         γ*dot(wit.point, vec - vec2) +
#         #         β*norm(deriv) ≤ 0
#         # ) # old
#     end
# end # old

function compute_vecs(vecsgen::VecsGenerator, G::Float64, H::Float64, solver)
    if iszero(vecsgen.nvec)
        return Vector{Float64}[], Inf
    end

    model = Model(solver)
    vecs = _add_vars!(model, vecsgen.nvar, vecsgen.nvec)
    r = @variable(model, upper_bound=2)

    for (i, wit) in vecsgen.witnesses_map
        _add_constrs_witness!(model, vecs, r, i, wit, vecsgen.ϵ, G, H)
    end

    @objective(model, Max, r)

    optimize!(model)

    if !(primal_status(model) == _RSC_(1) &&
            termination_status(model) == _TSC_(1))
        error(string(
            "Generator: not optimal: ",
            primal_status(model), " ",
            dual_status(model), " ",
            termination_status(model)
        ))
    end

    return [value.(vec) for vec in vecs], value(r)
end

function compute_vecs(vecsgen::VecsGenerator, solver)
    vecsopt = Vector{Float64}[]
    ropt = -Inf
    for (k, G) in enumerate(vecsgen.Gs)
        r = vecsgen.rs[k]
        r < ropt && continue
        vecs, r = compute_vecs(vecsgen, G, 2*G, solver)
        vecsgen.rs[k] = r
        if r > ropt
            ropt = r
            vecsopt = vecs
        end
    end
    if isinf(ropt)
        error(string("Generator: inf radius: ", ropt))
    end
    return vecsopt, ropt
end