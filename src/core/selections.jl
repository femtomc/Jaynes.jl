abstract type Selection end
abstract type ConstrainedSelection <: Selection end
abstract type UnconstrainedSelection <: Selection end

struct HierarchicalUnconstrainedSelection <: UnconstrainedSelection
    map::Dict{Address, HierarchicalUnconstrainedSelection}
end

struct HierarchicalConstrainedSelection <: ConstrainedSelection
    map::Dict{Address, HierarchicalConstrainedSelection}
    constraints::Dict{Address, Any}
    ConstrainedSelection(d::Dict{Union{Symbol, Pair}, T}) where T = new(collect(keys(d)), d)
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
