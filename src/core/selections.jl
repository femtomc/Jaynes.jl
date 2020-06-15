# Address selections are used by many different contexts.

abstract type Selection end

struct UnconstrainedSelection <: Selection
    map::Dict{Address, UnconstrainedSelection}
end

struct ConstrainedSelection <: Selection
    map::Dict{Address, ConstrainedSelection}
    constraints::Dict{Address, Any}
    ConstrainedSelection(d::Dict{Union{Symbol, Pair}, T}) where T = new(collect(keys(d)), d)
end

# Mutable - represents a set of learnable parameters in each call.
mutable struct LearnableUnconstrainedSelection <: Selection
    map::Dict{Address, LearnableUnconstrainedSelection}
    trainable::Dict{Address, Union{Number, AbstractArray}}
    gradients::Dict{Address, Vector{SiteGradients}}
    parents::Dict{Address, Vector{Address}}
    tracker::IdDict{Union{Number, AbstractArray}, Address}
end

mutable struct LearnableConstrainedSelection <: Selection
    map::Dict{Address, LearnableConstrainedSelection}
    trainable::Dict{Address, Union{Number, AbstractArray}}
    gradients::Dict{Address, Vector{SiteGradients}}
    parents::Dict{Address, Vector{Address}}
    tracker::IdDict{Union{Number, AbstractArray}, Address}
    constraints::Dict{Address, Any}
    LearnableConstrainedSelection(d::Dict{Union{Symbol, Pair}, T}) where T = new(collect(keys(d)), d)
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
