# Map hierarchy.
abstract type AddressMap{K} end

abstract type Leaf{K} <: AddressMap{K} end

struct Empty <: Leaf{Empty} end

struct Value{K} <: Leaf{Value}
    val::K
end

isempty(v::Value) = false
unwrap(v::Value) = v.val

abstract type Selection <: Leaf{Selection} end

struct DynamicMap{K} <: AddressMap{K}
    tree::Dict{Any, AddressMap{<:K}}
    DynamicMap{K}() where K = new{K}(Dict{Any, AddressMap{K}}())
end

push!(dm::DynamicMap{K}, addr, v::AddressMap{<:K}) where K = dm.tree[addr] = v

haskey(dm::DynamicMap, addr) = haskey(dm.tree, addr) && !isempty(dm.tree[addr])

function getindex(dm::DynamicMap, addr)
    val = dm.tree[addr]
    isempty(val) ? Empty() : unwrap(val)
end

function get(dm::DynamicMap{K}, addr, val) where K
    haskey(dm, addr) && return getindex(dm, addr)
    return val
end

# Traces
include("traces.jl")

# Selections
include("selections.jl")
