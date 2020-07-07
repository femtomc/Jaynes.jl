import Base: haskey, getindex, push!, merge!, union, isempty

# Address selections are used by many different contexts. 
# This is essentially a query language for addresses within a particular method body.
abstract type Selection end

# ------------ Lightweight visitor ------------ #

struct Visitor <: Selection
    tree::Dict{Address, Visitor}
    addrs::Vector{Address}
    Visitor() = new(Dict{Address, Visitor}(), Address[])
end

push!(vs::Visitor, addr::Address) = push!(vs.addrs, addr)

function visit!(vs::Visitor, addr::Address)
    addr in vs.addrs && error("VisitorError (visit!): already visited this address.")
    push!(vs, addr)
end
function set_sub!(vs::Visitor, addr::Address, sub::Visitor)
    haskey(vs.tree, addr) && error("VisitorError (set_sub!): already visited this address.")
    vs.tree[addr] = sub
end
function has_sub(vs::Visitor, addr::Address)
    return haskey(vs.tree, addr)
end
function get_sub(vs::Visitor, addr::Address)
    haskey(vs.tree, addr) && return vs.tree[addr]
    error("VisitorError (get_sub): sub not defined at $addr.")
end
function has_query(vs::Visitor, addr::Address)
    return addr in vs.addrs
end
isempty(vs::Visitor) = isempty(vs.tree) && isempty(vs.addrs)

# ------------ Constrained and unconstrained selections ------------ #

abstract type ConstrainedSelection end
abstract type UnconstrainedSelection end
abstract type SelectQuery <: Selection end
abstract type ConstrainedSelectQuery <: SelectQuery end
abstract type UnconstrainedSelectQuery <: SelectQuery end

# Empty select query.
struct EmptySelectQuery <: SelectQuery end

# Constraints to direct addresses.
struct ConstrainedSelectByAddress <: ConstrainedSelectQuery
    query::Dict{Address, Any}
    ConstrainedSelectByAddress() = new(Dict{Address, Any}())
    ConstrainedSelectByAddress(d::Dict{Address, Any}) = new(d)
end
has_query(csa::ConstrainedSelectByAddress, addr) = haskey(csa.query, addr)
get_query(csa::ConstrainedSelectByAddress, addr) = getindex(csa.query, addr)
isempty(csa::ConstrainedSelectByAddress) = isempty(csa.query)

# Selection to direct addresses.
struct UnconstrainedSelectByAddress <: UnconstrainedSelectQuery
    query::Vector{Address}
    UnconstrainedSelectByAddress() = new(Address[])
end
has_query(csa::UnconstrainedSelectByAddress, addr) = addr in csa.query
isempty(csa::UnconstrainedSelectByAddress) = isempty(csa.query)

# Constrain anywhere.
struct ConstrainedAnywhereSelection{T <: ConstrainedSelectQuery} <: ConstrainedSelection
    query::T
    ConstrainedAnywhereSelection(obs::Vector{Tuple{T, K}}) where {T <: Any, K} = new{ConstrainedSelectByAddress}(ConstrainedSelectByAddress(Dict{Address, Any}(obs)))
    ConstrainedAnywhereSelection(obs::Tuple{T, K}...) where {T <: Any, K} = new{ConstrainedSelectByAddress}(ConstrainedSelectByAddress(Dict{Address, Any}(collect(obs))))
end
has_query(cas::ConstrainedAnywhereSelection, addr) = has_query(cas.query, addr)
get_query(cas::ConstrainedAnywhereSelection, addr) = get_query(cas.query, addr)
get_sub(cas::ConstrainedAnywhereSelection, addr) = cas
isempty(cas::ConstrainedAnywhereSelection) = isempty(cas.query)

# Constrain in call hierarchy.
struct ConstrainedHierarchicalSelection{T <: ConstrainedSelectQuery} <: ConstrainedSelection
    tree::Dict{Union{Int, Address}, ConstrainedSelection}
    query::T
    ConstrainedHierarchicalSelection() = new{ConstrainedSelectByAddress}(Dict{Union{Int, Address}, ConstrainedHierarchicalSelection}(), ConstrainedSelectByAddress())
end

function get_sub(chs::ConstrainedHierarchicalSelection, addr)
    haskey(chs.tree, addr) && return chs.tree[addr]
    return ConstrainedHierarchicalSelection()
end
function get_sub(chs::ConstrainedHierarchicalSelection, addr::Pair)
    haskey(chs.tree, addr[1]) && return get_sub(chs.tree[addr[1]], addr[2])
    return ConstrainedHierarchicalSelection()
end
has_query(chs::ConstrainedHierarchicalSelection, addr) = has_query(chs.query, addr)
get_query(chs::ConstrainedHierarchicalSelection, addr) = get_query(chs.query, addr)
isempty(chs::ConstrainedHierarchicalSelection) = isempty(chs.tree) && isempty(chs.query)

struct UnconstrainedHierarchicalSelection{T <: UnconstrainedSelectQuery} <: UnconstrainedSelection
    tree::Dict{Address, UnconstrainedHierarchicalSelection}
    query::T
    UnconstrainedHierarchicalSelection() = new{UnconstrainedSelectByAddress}(Dict{Address, UnconstrainedHierarchicalSelection}(), UnconstrainedSelectByAddress())
end
function get_sub(uhs::UnconstrainedHierarchicalSelection, addr)
    haskey(uhs.tree, addr) && return uhs.tree[addr]
    return UnconstrainedHierarchicalSelection()
end
function get_sub(uhs::UnconstrainedHierarchicalSelection, addr::Pair)
    haskey(uhs.tree, addr[1]) && return get_sub(uhs.tree[addr[1]], addr[2])
    return UnconstrainedHierarchicalSelection()
end
has_query(uhs::UnconstrainedHierarchicalSelection, addr) = has_query(uhs.query, addr)
isempty(uhs::UnconstrainedHierarchicalSelection) = isempty(uhs.tree) && isempty(uhs.query)

# Union of constraints.
struct ConstrainedUnionSelection <: ConstrainedSelection
    query::Vector{ConstrainedSelection}
end
function has_query(cus::ConstrainedUnionSelection, addr)
    for q in cus.query
        has_query(q, addr) && return true
    end
    return false
end
function get_query(cus::ConstrainedUnionSelection, addr)
    for q in cus.query
        has_query(q, addr) && return get_query(q, addr)
    end
    error("ConstrainedUnionSelection (get_query): query not defined for $addr.")
end
function get_sub(cus::ConstrainedUnionSelection, addr)
    return ConstrainedUnionSelection(map(cus.query) do q
                                         get_sub(q, addr)
                                     end)
end
isempty(cus::ConstrainedUnionSelection) = isempty(cus.query)

struct UnconstrainedUnionSelection <: UnconstrainedSelection
    query::Vector{UnconstrainedSelection}
end
function has_query(uus::UnconstrainedUnionSelection, addr)
    for q in uus.query
        has_query(q, addr) && return true
    end
    return false
end
function get_query(uus::UnconstrainedUnionSelection, addr)
    for q in uus.query
        has_query(q, addr) && return get_query(q, addr)
    end
    error("ConstrainedUnionSelection (get_query): query not defined for $addr.")
end
function get_sub(uus::UnconstrainedUnionSelection, addr)
    return UnconstrainedUnionSelection(map(uus.query) do q
                                           get_sub(q, addr)
                                       end)
end
isempty(uus::UnconstrainedUnionSelection) = isempty(uus.query)

# Set operations.
union(a::ConstrainedSelection...) = ConstrainedUnionSelection([a...])
function intersection(a::ConstrainedSelection...) end

# Builders.
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
    for k in keys(subtrace.calls)
        site_push!(subchs, k, subtrace.calls[k])
    end
    for k in keys(subtrace.choices)
        site_push!(subchs, k, subtrace.calls[k])
    end
    chs.tree[addr] = subchs
end
function push!(chs::ConstrainedHierarchicalSelection, tr::HierarchicalTrace)
    for k in keys(tr.calls)
        site_push!(chs, k, tr.calls[k])
    end
    for k in keys(tr.choices)
        site_push!(chs, k, tr.choices[k])
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

# Compare.
addresses(csa::ConstrainedSelectByAddress) = keys(csa.query)
addresses(usa::UnconstrainedSelectByAddress) = usa.query
function compare(chs::ConstrainedHierarchicalSelection, v::Visitor)::Bool
    for addr in addresses(chs.query)
        addr in v.addrs || return false
    end
    for addr in keys(chs.tree)
        haskey(v.tree, addr) || return false
        compare(chs.tree[addr], v.tree[addr]) || return false
    end
    return true
end

# Merge observations and a choice map.
function merge(tr::HierarchicalTrace,
               obs::ConstrainedHierarchicalSelection)
    tr_selection = selection(tr)
    merge!(tr_selection, obs)
    return tr_selection
end
