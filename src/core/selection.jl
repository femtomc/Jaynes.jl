abstract type Selection end

struct UnconstrainedSelection <: Selection
    addresses::Vector{Address}
end

struct ConstrainedSelection{T} <: Selection
    addresses::Vector{Address}
    constraints::Dict{Address, T}
    ConstrainedSelection(d::Dict{Address, T}) where T = new{T}(collect(keys(d)), d)
end

import Base: haskey, setindex!, getindex
Base.haskey(s::Selection, key::Address) = key in s.addresses
Base.getindex(s::ConstrainedSelection, key::Address) = s.constraints[key]
function Base.setindex!(s::ConstrainedSelection{T}, key::Address, val::T) where T
    s.constraints[key] = val
end
