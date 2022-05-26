module Polyhedra

using LinearAlgebra

_VT_ = Vector{Float64}
_MT_ = Matrix{Float64}

struct Supp
    a::_VT_
end

Base.in(x, s::Supp) = dot(s.a, x) ≤ 0

struct Cone
    supps::Vector{Supp}
end

Cone() = Cone(_VT_[])
    
function add_supp!(c::Cone, s::Supp)
    push!(c.supps, s)
end

function add_supp!(c::Cone, a::_VT_)
    add_supp!(c, Supp(a))
end

Base.in(x, c::Cone) = all(s -> x ∈ s, c.supps)

struct Halfspace
    a::_VT_
    β::Float64
end

Base.in(x, h::Halfspace) = dot(h.a, x) + h.β ≤ 0

struct Polyhedron
    halfspaces::Vector{Halfspace}
end

Polyhedron() = Polyhedron(Halfspace[])

function add_halfspace!(p::Polyhedron, h::Halfspace)
    push!(p.halfspaces, h)    
end

function add_halfspace!(p::Polyhedron, a::_VT_, β::Float64)
    add_halfspace!(p, Halfspace(a, β))
end

Base.in(x, p::Polyhedron) = all(h -> x ∈ h, p.halfspaces)

end # module