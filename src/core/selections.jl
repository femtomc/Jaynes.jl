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
function push!(sel::ConstrainedSelectByAddress, addr::Symbol, val)
    sel.select[addr] = val
end
function push!(sel::UnconstrainedSelectByAddress, addr::Pair{Symbol, Int64})
    push!(sel.select, addr)
end
function push!(sel::ConstrainedSelectByAddress, addr::Pair{Symbol, Int64}, val)
    sel.select[addr] = val
end
function push!(sel::UnconstrainedHierarchicalSelection, addr::Symbol)
    push!(sel.select, addr)
end
function push!(sel::ConstrainedHierarchicalSelection, addr::Symbol, val)
    push!(sel.select, addr, val)
end
function push!(sel::UnconstrainedHierarchicalSelection, addr::Pair{Symbol, Int64})
    push!(sel.select, addr)
end
function push!(sel::ConstrainedHierarchicalSelection, addr::Pair{Symbol, Int64}, val)
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
function push!(sel::ConstrainedHierarchicalSelection, addr::Pair, val)
    if !(haskey(sel.tree, addr[1]))
        new = ConstrainedHierarchicalSelection()
        push!(new, addr[2], val)
        sel.tree[addr[1]] = new
    else
        push!(sel[addr[1]], addr[2], val)
    end
end
function UnconstrainedHierarchicalSelection(a::Vector{K}) where K <: Union{Symbol, Pair}
    top = UnconstrainedHierarchicalSelection()
    for addr in a
        push!(top, addr)
    end
    return top
end
function ConstrainedHierarchicalSelection(a::Vector{Tuple{K, T}}) where {T, K <: Union{Symbol, Pair}}
    top = ConstrainedHierarchicalSelection()
    for (addr, val) in a
        push!(top, addr, val)
    end
    return top
end

# Merges two selections, overwriting the first.
import Base.merge!
function merge!(sel1::ConstrainedSelectByAddress,
                sel2::ConstrainedSelectByAddress)
    Base.merge!(sel1.select, sel2.select)
end

function merge!(sel1::ConstrainedHierarchicalSelection,
                sel2::ConstrainedHierarchicalSelection)
    merge!(sel1.select, sel2.select)
    for k in keys(sel2.tree)
        if haskey(sel1.tree, k)
            merge!(sel1.tree[k], sel2.select[k])
        else
            sel1.tree[k] = sel2.select[k]
        end
    end
end

# Produces a hierarchical selection from a trace.
function site_push!(chs::ConstrainedHierarchicalSelection, addr::Address, cs::ChoiceSite)
    push!(chs, addr, cs.val)
end
function site_push!(chs::ConstrainedHierarchicalSelection, addr::Address, cs::CallSite)
    subtrace = cs.trace
    for k in keys(subtrace.chm)
        push!(chs, k, subtrace.chm[k])
    end
end
function push!(chs::ConstrainedHierarchicalSelection, tr::HierarchicalTrace)
    for k in keys(tr.chm)
        site_push!(chs, k, tr.chm[k])
    end
end
function chm(tr::HierarchicalTrace)
    top = ConstrainedHierarchicalSelection()
    push!(top, tr)
    return top
end

# Generic builders.
function selection(a::Vector{Tuple{K, T}})  where {T, K <: Union{Symbol, Pair}}
    return ConstrainedHierarchicalSelection(a)
end
function selection(a::Vector{K}) where K <: Union{Symbol, Pair}
    return UnconstrainedHierarchicalSelection(a)
end
function selection(a::Tuple{K, T}...) where {T, K <: Union{Symbol, Pair}}
    observations = Vector{Tuple{K, T}}(collect(a))
    return ConstrainedHierarchicalSelection(observations)
end

