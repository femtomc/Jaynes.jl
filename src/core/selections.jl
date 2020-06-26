import Base: haskey, getindex, push!, merge!, union

# Address selections are used by many different contexts. 
# This is essentially a query language for addresses within a particular method body.
abstract type Selection end

# Lightweight visitor - used by regenerate contexts.
struct VisitedSelection <: Selection
    tree::Dict{Address, VisitedSelection}
    addrs::Vector{Address}
    VisitedSelection() = new(Dict{Address, VisitedSelection}(), Address[])
end
push!(vs::VisitedSelection, addr::Address) = push!(vs.addrs, addr)

abstract type ConstrainedSelection end
abstract type UnconstrainedSelection end
abstract type SelectQuery <: Selection end
abstract type ConstrainedSelectQuery <: SelectQuery end
abstract type UnconstrainedSelectQuery <: SelectQuery end

struct ConstrainedSelectByAddress <: ConstrainedSelectQuery
    query::Dict{Address, Any}
    ConstrainedSelectByAddress() = new(Dict{Address, Any}())
    ConstrainedSelectByAddress(d::Dict{Address, Any}) = new(d)
end


struct UnconstrainedSelectByAddress <: UnconstrainedSelectQuery
    query::Vector{Address}
    UnconstrainedSelectByAddress() = new(Address[])
end


struct ConstrainedHierarchicalSelection{T <: ConstrainedSelectQuery} <: ConstrainedSelection
    tree::Dict{Address, ConstrainedHierarchicalSelection}
    query::T
    ConstrainedHierarchicalSelection() = new{ConstrainedSelectByAddress}(Dict{Address, ConstrainedHierarchicalSelection}(), ConstrainedSelectByAddress())
end

struct ConstrainedAnywhereSelection{T <: ConstrainedSelectQuery} <: ConstrainedSelection
    query::T
    ConstrainedAnywhereSelection(obs::Vector{Tuple{T, K}}) where {T <: Any, K} = new{ConstrainedSelectByAddress}(ConstrainedSelectByAddress(Dict{Address, Any}(obs)))
    ConstrainedAnywhereSelection(obs::Tuple{T, K}...) where {T <: Any, K} = new{ConstrainedSelectByAddress}(ConstrainedSelectByAddress(Dict{Address, Any}(collect(obs))))
end

struct ConstrainedUnionSelection <: ConstrainedSelection
    query::Vector{ConstrainedSelection}
end

struct UnconstrainedUnionSelection <: UnconstrainedSelection
    query::Vector{UnconstrainedSelection}
end

# Set operations.
union(a::ConstrainedSelection...) = ConstrainedUnionSelection([a...])

struct UnconstrainedHierarchicalSelection{T <: UnconstrainedSelectQuery} <: UnconstrainedSelection
    tree::Dict{Address, UnconstrainedHierarchicalSelection}
    query::T
    UnconstrainedHierarchicalSelection() = new{UnconstrainedSelectByAddress}(Dict{Address, UnconstrainedHierarchicalSelection}(), UnconstrainedSelectByAddress())
end

Base.haskey(usa::UnconstrainedSelectByAddress, addr::Address) = haskey(usa.query, addr)
Base.haskey(csa::ConstrainedSelectByAddress, addr::Address) = haskey(csa.query, addr)
Base.haskey(usa::UnconstrainedSelectByAddress, addr::Address) = addr in usa.query
Base.haskey(chs::ConstrainedHierarchicalSelection, addr::Address) = haskey(chs.tree, addr)
function Base.haskey(selections::Vector{ConstrainedSelection}, addr::T) where T <: Address
    for sel in selections
        if haskey(sel, addr)
            return true
        end
    end
    return false
end
Base.haskey(sel::ConstrainedAnywhereSelection, addr::T) where T <: Address = haskey(sel.query, addr)
Base.haskey(hs::UnconstrainedHierarchicalSelection, addr::Address) = haskey(hs.query, addr)

# Can't appear outside of a higher-level wrapper.
Base.getindex(csa::ConstrainedSelectByAddress, addr::Address) = csa.query[addr]

# Higher level wrappers.
Base.getindex(chs::ConstrainedHierarchicalSelection, addr::Address) = getindex(chs.tree, addr)
Base.getindex(chs::ConstrainedAnywhereSelection, addr::Address) = chs

unwrap(sel::ConstrainedAnywhereSelection, addr::Address) = sel
function unwrap(chs::ConstrainedHierarchicalSelection, addr::Address)
    if haskey(chs.tree, addr)
        chs.tree[addr]
    else
        ConstrainedHierarchicalSelection()
    end
end
function Base.getindex(us::ConstrainedUnionSelection, addr::Address)
    ConstrainedUnionSelection(map(us.query) do sel
                                  unwrap(sel, addr)
                              end)
end

# Handles queries to unions.
function Base.getindex(selections::Vector{ConstrainedSelection}, addr::Address)
    for sel in selections
        if haskey(sel, addr)
            return getindex(sel.query, addr)
        end
    end
end

# Builder.
function push!(sel::UnconstrainedSelectByAddress, addr::Symbol)
    push!(sel.query, addr)
end
function push!(sel::ConstrainedSelectByAddress, addr::Symbol, val)
    sel.query[addr] = val
end
function push!(sel::UnconstrainedSelectByAddress, addr::Pair{Symbol, Int64})
    push!(sel.query, addr)
end
function push!(sel::ConstrainedSelectByAddress, addr::Pair{Symbol, Int64}, val)
    sel.query[addr] = val
end
function push!(sel::UnconstrainedHierarchicalSelection, addr::Symbol)
    push!(sel.query, addr)
end
function push!(sel::ConstrainedHierarchicalSelection, addr::Symbol, val)
    push!(sel.query, addr, val)
end
function push!(sel::UnconstrainedHierarchicalSelection, addr::Pair{Symbol, Int64})
    push!(sel.query, addr)
end
function push!(sel::ConstrainedHierarchicalSelection, addr::Pair{Symbol, Int64}, val)
    push!(sel.query, addr, val)
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
function merge!(sel1::ConstrainedSelectByAddress,
                sel2::ConstrainedSelectByAddress)
    Base.merge!(sel1.query, sel2.query)
end

function merge!(sel1::ConstrainedHierarchicalSelection,
                sel2::ConstrainedHierarchicalSelection)
    merge!(sel1.query, sel2.query)
    for k in keys(sel2.tree)
        if haskey(sel1.tree, k)
            merge!(sel1.tree[k], sel2.query[k])
        else
            sel1.tree[k] = sel2.query[k]
        end
    end
end

# Produces a hierarchical selection from a trace.
function site_push!(chs::ConstrainedHierarchicalSelection, addr::Address, cs::ChoiceSite)
    push!(chs, addr, cs.val)
end
function site_push!(chs::ConstrainedHierarchicalSelection, addr::Address, cs::CallSite)
    subtrace = cs.trace
    subchs = ConstrainedHierarchicalSelection()
    for k in keys(subtrace.chm)
        site_push!(subchs, k, subtrace.chm[k])
    end
    chs.tree[addr] = subchs
end
function push!(chs::ConstrainedHierarchicalSelection, tr::HierarchicalTrace)
    for k in keys(tr.chm)
        site_push!(chs, k, tr.chm[k])
    end
end
function selection(tr::HierarchicalTrace)
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
function selection(a::Address...)
    observations = Vector{Address}(collect(a))
    return UnconstrainedHierarchicalSelection(observations)
end

# Get addresses.
addresses(csa::ConstrainedSelectByAddress) = keys(csa.query)
addresses(usa::UnconstrainedSelectByAddress) = usa.query

# Compare.
function compare(chs::ConstrainedHierarchicalSelection, v::VisitedSelection)::Bool
    for addr in addresses(chs.query)
        addr in v.addrs || return false
    end
    for addr in keys(chs.tree)
        haskey(v.tree, addr) || return false
        compare(chs.tree[addr], v.tree[addr]) || return false
    end
    return true
end
