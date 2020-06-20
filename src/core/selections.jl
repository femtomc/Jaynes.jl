# Address selections are used by many different contexts. 
# This is essentially a query language for addresses within a particular method body.
abstract type Selection end

abstract type SelectQuery <: Selection end
struct EmptySelection <: SelectQuery end

abstract type ConstrainedSelectQuery <: SelectQuery end
abstract type UnconstrainedSelectQuery <: SelectQuery end
struct SelectAll <: UnconstrainedSelectQuery end

struct ConstrainedSelectByAddress <: ConstrainedSelectQuery
    select::Dict{Address, Any}
end
struct UnconstrainedSelectByAddress <: UnconstrainedSelectQuery
    select::Vector{Address}
end

# Convenience - EmptySelection is allowed for both sub-structures.
const Unconstrained = Union{UnconstrainedSelectQuery, EmptySelection}
const Constrained = Union{ConstrainedSelectQuery, EmptySelection}

struct ConstrainedHierarchicalSelection{T <: Constrained} <: Selection
    tree::Dict{Address, ConstrainedHierarchicalSelection}
    select::T
    ConstrainedHierarchicalSelection() = new{EmptySelection}(Dict{Address, ConstrainedHierarchicalSelection}(), EmptySelection())
end

struct UnconstrainedHierarchicalSelection{T <: Unconstrained} <: Selection
    tree::Dict{Address, UnconstrainedHierarchicalSelection}
    select::T
    UnconstrainedHierarchicalSelection() = new{EmptySelection}(Dict{Address, UnconstrainedHierarchicalSelection}(), EmptySelection())
end

function UnconstrainedHierarchicalSelection(a::Vector{Tuple{Union{Symbol, Pair}}})
    isempty(a) && return UnconstrainedHierarchicalSelection()
    call_addrs, levels = create_levels(a)
    ks = sort(collect(keys(levels)))
    if !(0 in ks)
        first = minimum(ks)
        top = build_to_first(first)
        for k in ks
            lvl_sel = UnconstrainedSelectByAddress(levels[k])
            top[k] = UnconstrainedHierarchicalSelection(Dict{Address, UnconstrainedHierarchicalSelection}(), lvl_sel)
        end
    else
        top = UnconstrainedHierarchicalSelection(Dict{Address, UnconstrainedHierarchicalSelection}(), UnconstrainedSelectByAddress(levels[0]))
        for k in ks
            lvl_sel = UnconstrainedSelectByAddress(levels[k])
            top[k] = UnconstrainedHierarchicalSelection(Dict{Address, UnconstrainedHierarchicalSelection}(), lvl_sel)
        end
    end
    return top
end

function ConstrainedHierarchicalSelection(a::Vector{Tuple{Union{Symbol, Pair}, Any}})
    isempty(a) && return ConstrainedHierarchicalSelection()
    call_addrs, levels = create_levels(a)
    ks = sort(collect(keys(levels)))
    if !(0 in ks)
        first = minimum(ks)
        top = build_to_first(first)
        for k in ks
            lvl_sel = ConstrainedSelectByAddress(levels[k])
            top[k] = ConstrainedHierarchicalSelection(Dict{Address, ConstrainedHierarchicalSelection}(), lvl_sel)
        end
    else
        top = ConstrainedHierarchicalSelection(Dict{Address, ConstrainedHierarchicalSelection}(), ConstrainedSelectByAddress(levels[0]))
        for k in ks
            lvl_sel = ConstrainedSelectByAddress(levels[k])
            top[k] = ConstrainedHierarchicalSelection(Dict{Address, ConstrainedHierarchicalSelection}(), lvl_sel)
        end
    end
    return top
end

# Base imports.
import Base: haskey
Base.haskey(::EmptySelection, addr::Address) = false
Base.haskey(::SelectAll, addr::Address) = true
Base.haskey(usa::UnconstrainedSelectByAddress, addr::Address) = addr in usa.select
Base.haskey(csa::ConstrainedSelectByAddress, addr::Address) = addr in keys(csa.select)
Base.haskey(hs::ConstrainedHierarchicalSelection, addr::Address) = haskey(hs.select, addr)
Base.haskey(hs::UnconstrainedHierarchicalSelection, addr::Address) = haskey(hs.select, addr)

# Builder.
function selection(a::Vector{Tuple{Union{Symbol, Pair}, Any}}) 
    return ConstrainedHierarchicalSelection(a)
end
function selection(a::Vector{Tuple{Union{Symbol, Pair}}})
    return UnconstrainedHierarchicalSelection(a)
end
