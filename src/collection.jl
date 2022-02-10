abstract type LinkedCollection{T} end

Base.eltype(::Type{<:LinkedCollection{T}}) where T = T

mutable struct Nil{T} <: LinkedCollection{T}
end

mutable struct Cons{T} <: LinkedCollection{T}
    head::T
    tail::LinkedCollection{T}
end

push(t::LinkedCollection{T}, h) where {T} = Cons{T}(h, t)

nil(T) = Nil{T}()

function linked_collection(::Type{T}, elements) where T
    coll = nil(T)
    for elem in elements
        coll = push(coll, elem)
    end
    return coll
end

Base.length(l::Nil) = 0

function Base.length(l::Cons)
    n = 0
    for i in l
        n += 1
    end
    return n
end

Base.iterate(::LinkedCollection, ::Nil) = nothing
function Base.iterate(l::LinkedCollection, state::Cons=l)
    state.head, state.tail
end