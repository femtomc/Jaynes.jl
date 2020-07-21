# ------------ Constrain in call hierarchy ------------ #

struct ConstrainedHierarchicalSelection{T <: ConstrainedSelectQuery} <: ConstrainedSelection
    tree::Dict{Union{Int, Address}, ConstrainedSelection}
    query::T
    ConstrainedHierarchicalSelection() = new{ConstrainedByAddress}(Dict{Union{Int, Address}, ConstrainedHierarchicalSelection}(), ConstrainedByAddress())
    ConstrainedHierarchicalSelection(csa::T) where T <: ConstrainedSelectQuery = new{T}(Dict{Union{Int, Address}, ConstrainedHierarchicalSelection}(), csa)
end
function get_sub(chs::ConstrainedHierarchicalSelection, addr)
    haskey(chs.tree, addr) && return chs.tree[addr]
    return ConstrainedEmptySelection()
end
function get_sub(chs::ConstrainedHierarchicalSelection, addr::Pair)
    haskey(chs.tree, addr[1]) && return get_sub(chs.tree[addr[1]], addr[2])
    return ConstrainedEmptySelection()
end
function set_sub!(chs::ConstrainedHierarchicalSelection, addr::Address, sub::K) where K <: ConstrainedSelection
    chs.tree[addr] = sub
end
has_query(chs::ConstrainedHierarchicalSelection, addr::T) where T <: Address = has_query(chs.query, addr)
has_query(chs::ConstrainedHierarchicalSelection, addr::Int) = has_query(chs.query, addr)
function has_query(chs::ConstrainedHierarchicalSelection, addr::Pair)
    if haskey(chs.tree, addr[1])
        return has_query(get_sub(chs, addr[1]), addr[2])
    end
    return false
end
dump_queries(chs::ConstrainedHierarchicalSelection) = dump_queries(chs.query)
get_query(chs::ConstrainedHierarchicalSelection, addr) = get_query(chs.query, addr)
isempty(chs::ConstrainedHierarchicalSelection) = isempty(chs.tree) && isempty(chs.query)

# Used to merge observations.
function merge!(sel1::ConstrainedHierarchicalSelection, sel2::ConstrainedHierarchicalSelection)
    merge!(sel1.query, sel2.query)
    for k in keys(sel2.tree)
        if haskey(sel1.tree, k)
            merge!(sel1.tree[k], sel2.query[k])
        else
            sel1.tree[k] = sel2.query[k]
        end
    end
end
function merge(cl::T, sel::ConstrainedHierarchicalSelection) where T <: CallSite
    cl_selection = get_selection(cl)
    merge!(cl_selection, sel)
    return cl_selection
end

# Used to build.
function push!(sel::ConstrainedHierarchicalSelection, addr::Symbol, val)
    push!(sel.query, addr, val)
end
function push!(sel::ConstrainedHierarchicalSelection, addr::Pair{Symbol, Int64}, val)
    push!(sel.query, addr, val)
end
function push!(sel::ConstrainedHierarchicalSelection, addr::Pair, val)
    if !(haskey(sel.tree, addr[1]))
        new = ConstrainedHierarchicalSelection()
        push!(new, addr[2], val)
        sel.tree[addr[1]] = new
    else
        push!(get_sub(sel, addr[1]), addr[2], val)
    end
end

# Used for functional filter querying.
function filter(k_fn::Function, v_fn::Function, chs::ConstrainedHierarchicalSelection) where T <: Address
    top = ConstrainedHierarchicalSelection(filter(k_fn, v_fn, chs.query))
    for (k, v) in chs.tree
        top.tree[k] = filter(k_fn, v_fn, v)
    end
    return top
end

# To and from arrays.
function fill_array!(val::T, arr::Vector{K}, f_ind::Int) where {K, T <: ConstrainedHierarchicalSelection}
    sorted_toplevel_keys = sort(collect(addresses(val.query)))
    sorted_tree_keys  = sort(collect(keys(val.tree)))
    idx = f_ind
    for k in sorted_toplevel_keys
        v = val.utility[k]
        n = fill_array!(v, arr, idx)
        idx += n
    end
    for k in sorted_tree_keys
        n = fill_array!(get_sub(val, k), arr, idx)
        idx += n
    end
    idx - f_ind
end

function from_array(schema::T, arr::Vector{K}, f_ind::Int) where {K, T <: ConstrainedHierarchicalSelection}
    sel = T()
    sorted_toplevel_keys = sort(collect(addresses(schema.query)))
    sorted_tree_keys  = sort(collect(keys(schema.tree)))
    idx = f_ind
    for k in sorted_toplevel_keys
        (n, v) = from_array(schema.utility[k], arr, idx)
        idx += n
        sel.utility[k] = v
    end
    for k in sorted_tree_keys
        (n, v) = from_array(get_sub(schema, k), arr, idx)
        idx += n
        sel.tree[k] = v
    end
    (idx - f_ind, sel)
end

# Used for pretty printing.
function collect!(par::T, addrs::Vector{Any}, chd::Dict{Any, Any}, chs::ConstrainedHierarchicalSelection) where T <: Any
    collect!(par, addrs, chd, chs.query)
    for (k, v) in chs.tree
        collect!(par => k, addrs, chd, v)
    end
end
function collect!(addrs::Vector{Any}, chd::Dict{Any, Any}, chs::ConstrainedHierarchicalSelection)
    collect!(addrs, chd, chs.query)
    for (k, v) in chs.tree
        collect!(k, addrs, chd, v)
    end
end

function collect(chs::ConstrainedHierarchicalSelection)
    addrs = Any[]
    chd = Dict{Any, Any}()
    collect!(addrs, chd, chs)
    return addrs, chd
end

# Pretty printing.
function Base.display(chs::ConstrainedHierarchicalSelection; show_values = false)
    println("  __________________________________\n")
    println("             Constrained\n")
    addrs, chd = collect(chs)
    if show_values
        for a in addrs
            println(" $(a) : $(chd[a])")
        end
    else
        for a in addrs
            println(" $(a)")
        end
    end
    println("  __________________________________\n")
end

# ------------ CHS from vectors ----------- #

function ConstrainedHierarchicalSelection(a::Vector{Tuple{K, T}}) where {T, K}
    top = ConstrainedHierarchicalSelection()
    for (addr, val) in a
        push!(top, addr, val)
    end
    return top
end
function ConstrainedHierarchicalSelection(a::Vector{Tuple{Int, T}}) where T
    v_sel = ConstrainedByAddress(Dict(a))
    ConstrainedHierarchicalSelection(v_sel)
end

# ------------ CHS from traces ------------ #

function site_push!(chs::ConstrainedHierarchicalSelection, addr::Address, cs::ChoiceSite)
    push!(chs, addr, cs.val)
end
function site_push!(chs::ConstrainedHierarchicalSelection, addr::Address, cs::HierarchicalCallSite)
    subtrace = cs.trace
    subchs = ConstrainedHierarchicalSelection()
    for k in keys(subtrace.calls)
        site_push!(subchs, k, subtrace.calls[k])
    end
    for k in keys(subtrace.choices)
        site_push!(subchs, k, subtrace.choices[k])
    end
    chs.tree[addr] = subchs
end
function site_push!(chs::ConstrainedHierarchicalSelection, addr::Address, cs::VectorizedCallSite)
    for (k, subtrace) in enumerate(cs.subtraces)
        subchs = ConstrainedHierarchicalSelection()
        for k in keys(subtrace.calls)
            site_push!(subchs, k, subtrace.calls[k])
        end
        for k in keys(subtrace.choices)
            site_push!(subchs, k, subtrace.choices[k])
        end
        chs.tree[addr] = subchs
    end
end
function push!(chs::ConstrainedHierarchicalSelection, tr::HierarchicalTrace)
    for k in keys(tr.calls)
        site_push!(chs, k, tr.calls[k])
    end
    for k in keys(tr.choices)
        site_push!(chs, k, tr.choices[k])
    end
end
function get_selection(tr::HierarchicalTrace)
    top = ConstrainedHierarchicalSelection()
    push!(top, tr)
    return top
end
function get_selection(cl::HierarchicalCallSite)
    top = ConstrainedHierarchicalSelection()
    push!(top, cl.trace)
    return top
end

# ------------ Unconstrained selection in call hierarchy ------------ #

struct UnconstrainedHierarchicalSelection{T <: UnconstrainedSelectQuery} <: UnconstrainedSelection
    tree::Dict{Union{Int, Address}, UnconstrainedHierarchicalSelection}
    query::T
    UnconstrainedHierarchicalSelection() = new{UnconstrainedByAddress}(Dict{Address, UnconstrainedHierarchicalSelection}(), UnconstrainedByAddress())
end
function get_sub(uhs::UnconstrainedHierarchicalSelection, addr)
    haskey(uhs.tree, addr) && return uhs.tree[addr]
    return UnconstrainedEmptySelection()
end
function get_sub(uhs::UnconstrainedHierarchicalSelection, addr::Pair)
    haskey(uhs.tree, addr[1]) && return get_sub(uhs.tree[addr[1]], addr[2])
    return UnconstrainedEmptySelection()
end
has_query(uhs::UnconstrainedHierarchicalSelection, addr) = has_query(uhs.query, addr)
dump_queries(uhs::UnconstrainedHierarchicalSelection) = dump_queries(uhs.query)
isempty(uhs::UnconstrainedHierarchicalSelection) = isempty(uhs.tree) && isempty(uhs.query)
function set_sub!(uhs::UnconstrainedHierarchicalSelection, addr::Address, sub::K) where K <: UnconstrainedSelection
    uhs.tree[addr] = sub
end

# Used to build.
function push!(sel::UnconstrainedHierarchicalSelection, addr::T) where T <: Address
    push!(sel.query, addr)
end

function push!(sel::UnconstrainedHierarchicalSelection, addr::Int)
    push!(sel.query, addr)
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

# ------------ UHS from vectors ------------ #

function UnconstrainedHierarchicalSelection(a::Vector{K}) where K <: Union{Symbol, Int, Pair}
    top = UnconstrainedHierarchicalSelection()
    for addr in a
        push!(top, addr)
    end
    return top
end

# Used in functional filter querying.
function filter(k_fn::Function, chs::UnconstrainedHierarchicalSelection) where T <: Address
    top = UnconstrainedHierarchicalSelection(filter(k_fn, chs.query))
    for (k, v) in chs.tree
        top.tree[k] = filter(k_fn, v)
    end
    return top
end

# Used in pretty printing.
function collect!(par::T, addrs::Vector{Any}, chs::UnconstrainedHierarchicalSelection) where T <: Any
    collect!(par, chs.query)
    for (k, v) in chs.tree
        collect!(par => k, addrs, v)
    end
end

function collect!(addrs::Vector{Any}, chs::UnconstrainedHierarchicalSelection)
    collect!(addrs, chs.query)
    for (k, v) in chs.tree
        collect!(k, addrs, v)
    end
end

function collect(chs::UnconstrainedHierarchicalSelection)
    addrs = Any[]
    collect!(addrs, chs)
    return addrs
end

# Pretty printing.
function Base.display(chs::UnconstrainedHierarchicalSelection)
    println("  __________________________________\n")
    println("              Selection\n")
    addrs = collect(chs)
    for a in addrs
        println(" $(a)")
    end
    println("  __________________________________\n")
end

