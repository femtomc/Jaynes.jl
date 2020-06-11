abstract type Selection end

struct EmptySelection <: Selection end

struct UnconstrainedSelection <: Selection
    addresses::Vector{Union{Symbol, Pair}}
end

struct ConstrainedSelection{T} <: Selection
    addresses::Vector{Union{Symbol, Pair}}
    constraints::Dict{Union{Symbol, Pair}, T}
    ConstrainedSelection(d::Dict{Union{Symbol, Pair}, T}) where T = new{T}(collect(keys(d)), d)
end

import Base: haskey, setindex!, getindex
Base.haskey(s::ConstrainedSelection, key::Union{Symbol, Pair}) = key in s.addresses
Base.haskey(s::UnconstrainedSelection, key::Union{Symbol, Pair}) = key in s.addresses
Base.haskey(s::EmptySelection, key::Union{Symbol, Pair}) = error("KeyError: instances of type EmptySelection have no addresses.")
Base.getindex(s::ConstrainedSelection, key::Union{Symbol, Pair}) = s.constraints[key]
function Base.setindex!(s::ConstrainedSelection{T}, key::Union{Symbol, Pair}, val::T) where T
    s.constraints[key] = val
end
