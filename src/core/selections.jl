# Address selections are used by many different contexts. 
# This is essentially a query language for addresses within a particular method body.
abstract type Selection end
abstract type SelectQuery <: Selection end
abstract type ConstrainedSelectQuery <: SelectQuery end
abstract type UnconstrainedSelectQuery <: SelectQuery end

struct ConstrainedSelectByAddress <: ConstrainedSelectQuery
    select::Dict{Address, Any}
    ConstrainedSelectByAddress() = new(Dict{Address, Any}())
end
struct UnconstrainedSelectByAddress <: UnconstrainedSelectQuery
    select::Vector{Address}
    UnconstrainedSelectByAddress() = new(Address[])
end

struct ConstrainedHierarchicalSelection{T <: ConstrainedSelectQuery} <: Selection
    tree::Dict{Address, ConstrainedHierarchicalSelection}
    select::T
    ConstrainedHierarchicalSelection() = new{ConstrainedSelectByAddress}(Dict{Address, ConstrainedHierarchicalSelection}(), ConstrainedSelectByAddress())
end

struct UnconstrainedHierarchicalSelection{T <: UnconstrainedSelectQuery} <: Selection
    tree::Dict{Address, UnconstrainedHierarchicalSelection}
    select::T
    UnconstrainedHierarchicalSelection() = new{UnconstrainedSelectByAddress}(Dict{Address, UnconstrainedHierarchicalSelection}(), UnconstrainedSelectByAddress())
end

import Base: haskey
Base.haskey(usa::UnconstrainedSelectByAddress, addr::Address) = haskey(usa.select, addr)
Base.haskey(csa::ConstrainedSelectByAddress, addr::Address) = haskey(csa.select, addr)
Base.haskey(hs::ConstrainedHierarchicalSelection, addr::Address) = haskey(hs.select, addr)
Base.haskey(hs::UnconstrainedHierarchicalSelection, addr::Address) = haskey(hs.select, addr)

import Base: getindex
Base.getindex(csa::ConstrainedSelectByAddress, addr::Address) = csa.select[addr]
Base.getindex(chs::ConstrainedHierarchicalSelection, addr::Address) = getindex(chs.select, addr)

# Builder.
import Base.push!
function push!(sel::UnconstrainedSelectByAddress, addr::Symbol)
    push!(sel.select, addr)
end
function push!(sel::ConstrainedSelectByAddress, addr::Symbol, val::T) where T
    sel.select[addr] = val
end
function push!(sel::UnconstrainedSelectByAddress, addr::Pair{Symbol, Int64})
    push!(sel.select, addr)
end
function push!(sel::ConstrainedSelectByAddress, addr::Pair{Symbol, Int64}, val::T) where T
    sel.select[addr] = val
end
function push!(sel::UnconstrainedHierarchicalSelection, addr::Symbol)
    push!(sel.select, addr)
end
function push!(sel::ConstrainedHierarchicalSelection, addr::Symbol, val::T) where T
    push!(sel.select, addr, val)
end
function push!(sel::UnconstrainedHierarchicalSelection, addr::Pair{Symbol, Int64})
    push!(sel.select, addr)
end
function push!(sel::ConstrainedHierarchicalSelection, addr::Pair{Symbol, Int64}, val::T) where T
    push!(sel.select, addr, val)
end
function push!(sel::UnconstrainedHierarchicalSelection, addr::Pair)
    if !(haskey(sel.tree, addr[1]))
        new = UnconstrainedHierarchicalSelection()
        push!(new, addr[2])
        sel.tree[addr[1]] = new
    else
        push!(sel[addr[1]], addr[2])
    end
end
function push!(sel::ConstrainedHierarchicalSelection, addr::Pair, val::T) where T
    if !(haskey(sel.tree, addr[1]))
        new = ConstrainedHierarchicalSelection()
        push!(new, addr[2], val)
        sel.tree[addr[1]] = new
    else
        push!(sel[addr[1]], addr[2], val)
    end
end
function UnconstrainedHierarchicalSelection(a::Vector{Union{Symbol, Pair}})
    top = UnconstrainedHierarchicalSelection()
    for addr in a
        push!(top, addr)
    end
    return top
end
function ConstrainedHierarchicalSelection(a::Vector{Tuple{Union{Symbol, Pair}, Any}})
    top = ConstrainedHierarchicalSelection()
    for (addr, val) in a
        push!(top, addr, val)
    end
    return top
end

function selection(a::Vector{Tuple{Union{Symbol, Pair}, Any}}) 
    return ConstrainedHierarchicalSelection(a)
end
function selection(a::Vector{Union{Symbol, Pair}})
    return UnconstrainedHierarchicalSelection(a)
end
