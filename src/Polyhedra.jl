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

Base.in(x, c::Cone) = all(s -> x ∈ s, c.supps)

end # module