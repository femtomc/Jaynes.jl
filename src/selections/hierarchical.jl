# ------------ Constrain in call hierarchy ------------ #

struct ConstrainedHierarchicalSelection{T <: ConstrainedSelectQuery} <: ConstrainedSelection
    tree::Dict{Address, ConstrainedSelection}
    query::T
    ConstrainedHierarchicalSelection() = new{ConstrainedByAddress}(Dict{Address, ConstrainedHierarchicalSelection}(), ConstrainedByAddress())
    ConstrainedHierarchicalSelection(csa::T) where T <: ConstrainedSelectQuery = new{T}(Dict{Address, ConstrainedHierarchicalSelection}(), csa)
end

has_sub(chs::ConstrainedHierarchicalSelection, addr::T) where T <: Address = haskey(chs.tree, addr)
has_sub(chs::ConstrainedHierarchicalSelection, addr::Tuple{}) = false
has_sub(chs::ConstrainedHierarchicalSelection, addr::Tuple{T}) where T <: Address = has_sub(chs, addr[1])
function has_sub(chs::ConstrainedHierarchicalSelection, addr::T) where T <: Tuple
    has_sub(chs, addr[1]) && has_sub(get_sub(chs, addr[1]), addr[2 : end])
end

function get_sub(chs::ConstrainedHierarchicalSelection, addr::T) where T <: Address
    haskey(chs.tree, addr) && return chs.tree[addr]
    return ConstrainedEmptySelection()
end
get_sub(chs::ConstrainedHierarchicalSelection, addr::Tuple{}) = ConstrainedEmptySelection()
get_sub(chs::ConstrainedHierarchicalSelection, addr::Tuple{T}) where T <: Address = get_sub(chs, addr[1])
function get_sub(chs::ConstrainedHierarchicalSelection, addr::T) where T <: Tuple
    haskey(chs.tree, addr[1]) && return get_sub(chs.tree[addr[1]], addr[2 : end])
    return ConstrainedEmptySelection()
end

set_sub!(chs::ConstrainedHierarchicalSelection, addr::T, sub::K) where {T <: Address, K <: ConstrainedSelection} = chs.tree[addr] = sub
set_sub!(chs::ConstrainedHierarchicalSelection, addr::Tuple{}, sub::K) where K <: ConstrainedSelection = nothing
set_sub!(chs::ConstrainedHierarchicalSelection, addr::Tuple{T}, sub::K) where {T <: Address, K <: ConstrainedSelection} = set_sub!(chs, addr[1], sub)
function set_sub!(chs::ConstrainedHierarchicalSelection, addr::T, sub::K) where {T <: Tuple, K <: ConstrainedSelection}
    has_sub(chs, addr[1]) && set_sub!(chs.tree[addr[1]], addr[2 : end], sub)
end

has_top(chs::ConstrainedHierarchicalSelection, addr::T) where T <: Address = has_top(chs.query, addr)
function has_top(chs::ConstrainedHierarchicalSelection, addr::Tuple{}) where T <: Address
    return false
end
function has_top(chs::ConstrainedHierarchicalSelection, addr::Tuple{T}) where T <: Address
    return has_top(chs.query, addr[1])
end
function has_top(chs::ConstrainedHierarchicalSelection, addr::T) where T <: Tuple
    has_sub(chs, addr[1]) && return has_top(get_sub(chs, addr[1]), addr[2 : end])
end

get_top(chs::ConstrainedHierarchicalSelection, addr::T) where T <: Address = get_top(chs.query, addr)
get_top(chs::ConstrainedHierarchicalSelection, addr::Tuple{}) = nothing
get_top(chs::ConstrainedHierarchicalSelection, addr::Tuple{T}) where T <: Address = get_top(chs, addr[1])
function get_top(chs::ConstrainedHierarchicalSelection, addr::T) where T <: Tuple
    has_sub(chs, addr[1]) && return get_top(chs.tree[addr[1]], addr[2 : end])
    error("ConstrainedHierarchicalSelection (get_top): invalid index at $addr.")
end

haskey(chs::ConstrainedHierarchicalSelection, addr::T) where T <: Address = has_top(chs, addr) || has_sub(chs, addr)
haskey(chs::ConstrainedHierarchicalSelection, addr::Tuple{}) = false
haskey(chs::ConstrainedHierarchicalSelection, addr::Tuple{T}) where T <: Address = haskey(chs, addr[1])
function haskey(chs::ConstrainedHierarchicalSelection, addr::T) where T <: Tuple
    has_sub(chs, addr[1]) && haskey(get_sub(chs, addr[1]), addr[2 : end])
end

function getindex(chs::ConstrainedHierarchicalSelection, addr...)
    has_top(chs, addr) && return get_top(chs, addr)
    has_sub(chs, addr) && return get_sub(chs, addr)
    error("ConstrainedHierarchicalSelection (getindex): no query or subselection at $addr.")
end

push_query!(toplevel::Set, k::K, q::T) where {K, T <: Address} = push!(toplevel, (k, q))
push_query!(toplevel::Set, k::K, q::T) where {K, T <: Tuple} = push!(toplevel, (k, q...))

function dump_queries(chs::ConstrainedHierarchicalSelection)
    toplevel = dump_queries(chs.query)
    for (k, v) in chs.tree
        q = dump_queries(v)
        for (t, l) in q
            push!(toplevel, (k, t...) => l)
        end
    end
    return toplevel
end

isempty(chs::ConstrainedHierarchicalSelection) = begin
    !isempty(chs.query) && return false
    for (k, v) in chs.tree
        !isempty(chs.tree[k]) && return false
    end
    return true
end

function ==(sel1::ConstrainedHierarchicalSelection, sel2::ConstrainedHierarchicalSelection)
    sel1.query == sel2.query || return false
    keys(sel1.tree) == keys(sel2.tree) || return false
    for (k, v) in sel1.tree
        v == get_sub(sel2, k) || return false
    end
    return true
end

# Used to merge observations.
function merge!(sel1::ConstrainedHierarchicalSelection, sel2::ConstrainedHierarchicalSelection)
    overlapped = merge!(sel1.query, sel2.query)
    for k in keys(sel2.tree)
        if haskey(sel1.tree, k)
            overlapped = overlapped || merge!(sel1.tree[k], sel2.tree[k])
        else
            sel1.tree[k] = sel2.tree[k]
        end
    end
    overlapped
end
function merge(cl::T, sel::ConstrainedHierarchicalSelection) where T <: CallSite
    cl_selection = get_selection(cl)
    overlapped = merge!(cl_selection, sel)
    return cl_selection, overlapped
end

# Used to build.
push!(sel::ConstrainedHierarchicalSelection, addr::T, val) where T <: Address = push!(sel.query, addr, val)
push!(sel::ConstrainedHierarchicalSelection, addr::Tuple{}, val) = nothing
push!(sel::ConstrainedHierarchicalSelection, addr::Tuple{T}, val) where T <: Address = push!(sel, addr[1], val)
function push!(sel::ConstrainedHierarchicalSelection, addr::T, val) where T <: Tuple
    if has_sub(sel, addr[1])
        push!(get_sub(sel, addr[1]), addr[2 : end], val)
    else
        new = ConstrainedHierarchicalSelection()
        push!(new, addr[2 : end], val)
        sel.tree[addr[1]] = new
    end
end

# Used for functional filter querying.
function filter(k_fn::Function, v_fn::Function, chs::ConstrainedHierarchicalSelection) where T <: Address
    top = ConstrainedHierarchicalSelection(filter(k_fn, v_fn, chs.query))
    for (k, v) in chs.tree
        top.tree[k] = filter(k_fn, v_fn, v)
    end
    isempty(top) && return ConstrainedEmptySelection()
    return top
end

# To and from arrays.
function fill_array!(val::T, arr::Vector{K}, f_ind::Int) where {K, T <: ConstrainedHierarchicalSelection}
    sorted_toplevel_keys = sort(collect(addresses(val.query)))
    sorted_tree_keys  = sort(collect(keys(val.tree)))
    idx = f_ind
    for k in sorted_toplevel_keys
        v = get_top(val, k)
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
    sel = ConstrainedHierarchicalSelection()
    sorted_toplevel_keys = sort(collect(addresses(schema.query)))
    sorted_tree_keys  = sort(collect(keys(schema.tree)))
    idx = f_ind
    for k in sorted_toplevel_keys
        (n, v) = from_array(get_top(schema, k), arr, idx)
        idx += n
        push!(sel, k, v)
    end
    for k in sorted_tree_keys
        (n, v) = from_array(get_sub(schema, k), arr, idx)
        idx += n
        sel.tree[k] = v
    end
    (idx - f_ind, sel)
end

# Used for pretty printing.
function collect!(args...) end
function collect!(par::T, addrs::Vector{Any}, chd::Dict{Any, Any}, chs::ConstrainedHierarchicalSelection) where T <: Tuple
    collect!(par, addrs, chd, chs.query)
    for (k, v) in chs.tree
        collect!((par..., k), addrs, chd, v)
    end
end
function collect!(addrs::Vector{Any}, chd::Dict{Any, Any}, chs::ConstrainedHierarchicalSelection)
    collect!(addrs, chd, chs.query)
    for (k, v) in chs.tree
        collect!((k, ), addrs, chd, v)
    end
end

function collect(chs::ConstrainedHierarchicalSelection)
    addrs = Any[]
    chd = Dict{Any, Any}()
    collect!(addrs, chd, chs)
    return addrs, chd
end

# Pretty printing.
function Base.display(chs::ConstrainedHierarchicalSelection; show_values = true)
    println("  __________________________________\n")
    println("             Selection\n")
    addrs, chd = collect(chs)
    if show_values
        for a in addrs
            println(" $(a) = $(chd[a])")
        end
    else
        for a in addrs
            println(" $(a)")
        end
    end
    println("  __________________________________\n")
end

# ------------ CHS from vectors ----------- #

function ConstrainedHierarchicalSelection(a::Vector{Pair{K, J}}) where {K <: Tuple, J}
    top = ConstrainedHierarchicalSelection()
    for (addr, val) in a
        push!(top, addr, val)
    end
    return top
end

# ------------ CHS from traces ------------ #

site_push!(chs::ConstrainedHierarchicalSelection, addr::T, cs::ChoiceSite) where T <: Address = push!(chs, addr, cs.val)
site_push!(chs::ConstrainedHierarchicalSelection, addr::T, cs::CallSite) where T <: Address = set_sub!(chs, addr, get_selection(cs))
function push!(chs::ConstrainedHierarchicalSelection, tr::VectorizedTrace)
    for (k, cs) in enumerate(tr.subrecords)
        site_push!(chs, k, cs)
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
function get_selection(tr::VectorizedTrace)
    top = ConstrainedHierarchicalSelection()
    push!(top, tr)
    return top
end
function get_selection(tr::HierarchicalTrace)
    top = ConstrainedHierarchicalSelection()
    push!(top, tr)
    return top
end
function get_selection(cl::VectorizedCallSite)
    top = ConstrainedHierarchicalSelection()
    push!(top, cl.trace)
    return top
end
function get_selection(cl::HierarchicalCallSite)
    top = ConstrainedHierarchicalSelection()
    push!(top, cl.trace)
    return top
end

# ------------ Unconstrained selection in call hierarchy ------------ #

struct UnconstrainedHierarchicalSelection{T <: UnconstrainedSelectQuery} <: UnconstrainedSelection
    tree::Dict{Address, UnconstrainedHierarchicalSelection}
    query::T
    UnconstrainedHierarchicalSelection() = new{UnconstrainedByAddress}(Dict{Address, UnconstrainedHierarchicalSelection}(), UnconstrainedByAddress())
end

function get_sub(uhs::UnconstrainedHierarchicalSelection, addr::T) where T <: Address
    haskey(uhs.tree, addr) && return uhs.tree[addr]
    return UnconstrainedEmptySelection()
end
get_sub(uhs::UnconstrainedHierarchicalSelection, addr::Tuple{}) = UnconstrainedEmptySelection()
get_sub(uhs::UnconstrainedHierarchicalSelection, addr::Tuple{T}) where T <: Address = get_sub(uhs, addr[1])
function get_sub(uhs::UnconstrainedHierarchicalSelection, addr::T) where T <: Tuple
    haskey(uhs.tree, addr[1]) && return get_sub(uhs.tree[addr[1]], addr[2 : end])
    return UnconstrainedEmptySelection()
end

has_top(uhs::UnconstrainedHierarchicalSelection, addr::T) where T <: Address = has_top(uhs.query, addr)
has_top(uhs::UnconstrainedHierarchicalSelection, addr::Tuple{}) where T <: Address = false
has_top(uhs::UnconstrainedHierarchicalSelection, addr::Tuple{T}) where T <: Address = has_top(uhs, addr[1])
function has_top(uhs::UnconstrainedHierarchicalSelection, addr::T) where T <: Tuple
    haskey(uhs.tree, addr[1]) && has_top(uhs.tree[addr[1]], addr[2 : end])
end

has_sub(chs::UnconstrainedHierarchicalSelection, addr::T) where T <: Address = haskey(chs.tree, addr)
has_sub(chs::UnconstrainedHierarchicalSelection, addr::Tuple{}) = false
has_sub(chs::UnconstrainedHierarchicalSelection, addr::Tuple{T}) where T <: Address = has_sub(chs, addr[1])
function has_sub(chs::UnconstrainedHierarchicalSelection, addr::T) where T <: Tuple
    has_sub(chs, addr[1]) && has_sub(get_sub(chs, addr[1]), addr[2 : end])
end

function set_sub!(uhs::UnconstrainedHierarchicalSelection, addr::T, sub::K) where {T <: Address, K <: UnconstrainedSelection}
    uhs.tree[addr] = sub
end
set_sub!(uhs::UnconstrainedHierarchicalSelection, addr::Tuple{}, sub::K) where K <: UnconstrainedSelection = nothing
function set_sub!(uhs::UnconstrainedHierarchicalSelection, addr::Tuple{T}, sub::K) where {T <: Address, K <: UnconstrainedSelection}
    set_sub!(uhs, addr[1], sub)
end
function set_sub!(uhs::UnconstrainedHierarchicalSelection, addr::T, sub::K) where {T <: Tuple, K <: UnconstrainedSelection}
    haskey(uhs.tree, addr[1]) && set_sub!(uhs.tree[1], addr[2 : end], sub)
end

function dump_queries(uhs::UnconstrainedHierarchicalSelection)
    toplevel = dump_queries(uhs.query)
    for (k, v) in uhs.tree
        toplevel = union(toplevel, dump_queries(v))
    end
    return toplevel
end

haskey(chs::UnconstrainedHierarchicalSelection, addr::T) where T <: Address = has_top(chs, addr) || has_sub(chs, addr)
haskey(chs::UnconstrainedHierarchicalSelection, addr::Tuple{}) = false
haskey(chs::UnconstrainedHierarchicalSelection, addr::Tuple{T}) where T <: Address = haskey(chs, addr[1])
function haskey(chs::UnconstrainedHierarchicalSelection, addr::T) where T <: Tuple
    has_sub(chs, addr[1]) && haskey(get_sub(chs, addr[1]), addr[2 : end])
end

isempty(uhs::UnconstrainedHierarchicalSelection) = begin
    isempty(uhs.query) || return false
    for (k, v) in uhs.tree
        isempty(uhs.tree[k]) || return false
    end
    return true
end

function ==(sel1::UnconstrainedHierarchicalSelection, sel2::UnconstrainedHierarchicalSelection)
    sel1.query == sel2.query || return false
    keys(sel1.tree) == keys(sel2.tree) || return false
    for (k, v) in sel1.tree
        v == get_sub(sel2, k) || return false
    end
    return true
end

# Used to merge observations.
function merge!(sel1::UnconstrainedHierarchicalSelection, sel2::UnconstrainedHierarchicalSelection)
    merge!(sel1.query, sel2.query)
    for k in keys(sel2.tree)
        if haskey(sel1.tree, k)
            merge!(sel1.tree[k], sel2.tree[k])
        else
            sel1.tree[k] = sel2.tree[k]
        end
    end
end
function merge(cl::T, sel::UnconstrainedHierarchicalSelection) where T <: CallSite
    cl_selection = get_selection(cl)
    un_sel = selection(map(collect(dump_queries(cl_selection))) do k
                           (k, )
                       end)
    merge!(un_sel, sel)
    return un_sel
end

# Used to build.
push!(sel::UnconstrainedHierarchicalSelection, addr::T) where T <: Address = push!(sel.query, addr)
push!(sel::UnconstrainedHierarchicalSelection, addr::Tuple{}) = nothing
push!(sel::UnconstrainedHierarchicalSelection, addr::Tuple{T}) where T <: Address = push!(sel, addr[1])
function push!(sel::UnconstrainedHierarchicalSelection, addr::Tuple)
    if haskey(sel.tree, addr[1])
        push!(get_sub(sel, addr[1]), addr[2 : end])
    else
        new = UnconstrainedHierarchicalSelection()
        push!(new, addr[2 : end])
        sel.tree[addr[1]] = new
    end
end

# ------------ UHS from vectors ------------ #

function UnconstrainedHierarchicalSelection(a::Vector{K}) where K <: Tuple
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
    isempty(top) && return UnconstrainedEmptySelection()
    return top
end

# Used in pretty printing.
function collect!(par::T, addrs::Vector{Any}, chs::UnconstrainedHierarchicalSelection) where T <: Any
    collect!(par, addrs, chs.query)
    for (k, v) in chs.tree
        collect!((par..., k), addrs, v)
    end
end

function collect!(addrs::Vector{Any}, chs::UnconstrainedHierarchicalSelection)
    collect!(addrs, chs.query)
    for (k, v) in chs.tree
        collect!((k, ), addrs, v)
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

