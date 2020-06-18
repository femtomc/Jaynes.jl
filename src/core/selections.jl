# Address selections are used by many different contexts.

abstract type Selection end

struct UnconstrainedSelection <: Selection
    map::Dict{Address, UnconstrainedSelection}
    learned::Dict{Address, Any}
    UnconstrainedSelection(d::Dict{Address, UnconstrainedSelection}) = new(d, Dict{Address, Any}())
end

struct ConstrainedSelection <: Selection
    map::Dict{Address, ConstrainedSelection}
    constraints::Dict{Address, Any}
    learned::Dict{Address, Any}
    ConstrainedSelection(d::Dict{Union{Symbol, Pair}, T}) where T = new(collect(keys(d)), d, Dict{Address, Any}())
end

struct EmptySelection <: Selection end

import Base: haskey, setindex!, getindex
Base.haskey(s::ConstrainedSelection, key::Union{Symbol, Pair}) = key in s.addresses
Base.haskey(s::UnconstrainedSelection, key::Union{Symbol, Pair}) = key in s.addresses
Base.haskey(s::EmptySelection, key::Union{Symbol, Pair}) = false
Base.getindex(s::ConstrainedSelection, key::Union{Symbol, Pair}) = s.constraints[key]
function Base.setindex!(s::ConstrainedSelection, key::Union{Symbol, Pair}, val::T) where T
    s.constraints[key] = val
end
