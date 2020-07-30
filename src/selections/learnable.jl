# ------------ Utility selection, grads and params ------------ #

abstract type UtilitySelection <: Selection end

function fill_array!(val::T, arr::Vector{K}, f_ind::Int) where {K, T <: UtilitySelection}
    sorted_toplevel_keys = sort(collect(keys(val.utility)))
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

function from_array(schema::T, arr::Vector{K}, f_ind::Int) where {K, T <: UtilitySelection}
    # Subtypes of UtilitySelection must support an empty constructor.
    sel = T()
    sorted_toplevel_keys = sort(collect(keys(schema.utility)))
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

# ------------ Gradients ------------ #

struct Gradients <: UtilitySelection
    tree::Dict{Address, Gradients}
    utility::Dict{Address, Any}
    Gradients() = new(Dict{Address, Gradients}(), Dict{Address, Any}())
    Gradients(tree, utility) = new(tree, utility)
end

has_top(ps::Gradients, addr::T) where T <: Address = haskey(ps.utility, addr)
function has_top(ps::Gradients, addr::T) where T <: Tuple
    isempty(addr) && return false
    length(addr) == 1 && return has_top(ps, addr[1])
    return has_top(ps.tree[addr[1]], addr[2 : end])
end
get_top(ps::Gradients, addr::T) where T <: Address = getindex(ps.utility, addr)
function get_top(ps::Gradients, addr::T) where T <: Tuple
    isempty(addr) && error("get_top: index error - tuple address is empty.")
    length(addr) == 1 && return get_top(ps, addr[1])
    return get_top(ps.tree[addr[1]], addr[2 : end])
end
has_sub(ps::Gradients, addr::T) where T <: Address = haskey(ps.tree, addr)
function has_sub(ps::Gradients, addr::T) where T <: Tuple
    isempty(addr) && return false
    length(addr) == 1 && return has_sub(ps, addr[1])
    return has_sub(ps.tree[addr[1]], addr[2 : end])
end
get_sub(ps::Gradients, addr::T) where T <: Address = getindex(ps.tree, addr)
function get_sub(ps::Gradients, addr::T) where T <: Tuple
    isempty(addr) && error("get_sub: index error - tuple address is empty.")
    length(addr) == 1 && return get_sub(ps, addr[1])
    return get_sub(ps.tree[addr[1]], addr[2 : end])
end

# ------------ Builders ------------ #

function push!(ps::Gradients, addr::T, val) where T <: Address
    has_top(ps, addr) && begin
        ps.utility[addr] += val
        return
    end
    ps.utility[addr] = val
end

function push!(ps::Gradients, addr::T, val) where T <: Tuple
    isempty(addr) && return
    length(addr) == 1 && push!(ps, addr[1], val)
    hd = addr[1]
    tl = addr[2 : end]
    if has_sub(ps, hd)
        push!(get_sub(ps, hd), tl, val)
    else
        sub = Gradients()
        push!(sub, tl, val)
        ps.tree[hd] = sub
    end
end

# ------------ Adjoint ------------ #

Zygote.@adjoint Gradients(tree, utility) = Gradients(tree, utility), s_grad -> (nothing, nothing)

# ------------ Combining two sets of gradients ------------ #

function merge(sel1::Gradients,
               sel2::Gradients)
    utility = merge(sel1.utility, sel2.utility)
    tree = Dict{Address, Gradients}()
    for k in setdiff(keys(sel2.tree), keys(sel1.tree))
        tree[k] = sel2.tree[k]
    end
    for k in setdiff(keys(sel1.tree), keys(sel2.tree))
        tree[k] = sel1.tree[k]
    end
    for k in intersect(keys(sel1.tree), keys(sel2.tree))
        tree[k] = merge(sel1.tree[k], sel2.tree[k])
    end
    return Gradients(tree, utility)
end

+(a_grads::Gradients, b_grads::Gradients) = merge(a_grads, b_grads)

Zygote.@adjoint merge(a_grads, b_grads) = merge(a_grads, b_grads), s_grad -> (nothing, nothing)

# ------------ Parameters, empty and learnable ------------ #

abstract type Parameters <: UtilitySelection end

struct EmptyParameters <: Parameters end

Parameters() = EmptyParameters()
has_top(np::EmptyParameters, addr) = false
get_top(np::EmptyParameters, addr) = error("(get_param) called on instance of EmptyParameters. No parameters!")
has_sub(np::EmptyParameters, addr) = false
get_sub(np::EmptyParameters, addr) = np

struct LearnableParameters <: Parameters
    tree::Dict{Address, LearnableParameters}
    utility::Dict{Address, Any}
    LearnableParameters() = new(Dict{Address, LearnableParameters}(), Dict{Address, Any}())
end

has_sub(ps::LearnableParameters, addr::T) where T <: Address = haskey(ps.tree, addr)
function has_sub(ps::LearnableParameters, addr::T) where T <: Tuple
    isempty(addr) && return false
    length(addr) == 1 && return has_sub(ps, addr[1])
    has_sub(ps, addr[1]) && has_sub(get_sub(ps, addr[1]), addr[2 : end])
end

get_sub(ps::LearnableParameters, addr::T) where T <: Address = getindex(ps.tree, addr)
function get_sub(ps::LearnableParameters, addr::T) where T <: Tuple
    isempty(addr) && return Parameters()
    length(addr) == 1 && return get_sub(ps, addr[1])
    return get_sub(ps.tree[addr[1]], addr[2 : end])
end

set_sub!(ps::LearnableParameters, addr::T, sub::LearnableParameters) where T <: Address = ps.tree[addr] = sub
function set_sub!(ps::LearnableParameters, addr::T, sub::LearnableParameters) where T <: Tuple
    isempty(addr) && return
    length(addr) == 1 && set_sub!(ps, addr[1], sub)
    has_sub(ps, addr[1]) && set_sub!(get_sub(ps, addr[1]), addr[2 : end], sub)
end

has_top(ps::LearnableParameters, addr::T) where T <: Address = haskey(ps.utility, addr)
function has_top(ps::LearnableParameters, addr::T) where T <: Tuple
    isempty(addr) && return false
    length(addr) == 1 && return has_top(ps, addr[1])
    has_sub(ps, addr[1]) && return has_top(get_sub(ps, addr[1]), addr[2 : end])
end

get_top(ps::LearnableParameters, addr::T) where T <: Address = getindex(ps.utility, addr)
function get_top(ps::LearnableParameters, addr::T) where T <: Tuple
    isempty(addr) && error("LearnableParameters (get_top): invalid index at $addr.")
    length(addr) == 1 && return get_top(ps, addr[1])
    get_top(ps.tree[addr[1]], addr[2 : end])
end

getindex(ps::LearnableParameters, addrs...) = get_top(ps, addrs)

# ------------ Builders ------------ #

function push!(ps::LearnableParameters, addr::T, val) where T <: Address
    ps.utility[addr] = val
end

function push!(ps::LearnableParameters, addr::T, val) where T <: Tuple
    isempty(addr) && return
    length(addr) == 1 && push!(ps, addr[1], val)
    hd = addr[1]
    tl = addr[2 : end]
    if has_sub(ps, hd)
        push!(get_sub(ps, hd), tl, val)
    else
        sub = LearnableParameters()
        push!(sub, tl, val)
        ps.tree[hd] = sub
    end
end

function parameters(arr::Array{Pair{T, K}}) where {T <: Tuple, K}
    top = LearnableParameters()
    map(arr) do (k, v)
        push!(top, k, v)
    end
    return top
end

function parameters(p::Pair{T, K}) where {T <: Symbol, K <: Parameters}
    top = LearnableParameters()
    set_sub!(top, p[1], p[2])
    return top
end

# ------------ Adjoint ------------ #

Zygote.@adjoint LearnableParameters(tree, utility) = LearnableParameters(tree, utility), s_grad -> (nothing, nothing)

# ------------ Merging two sets of parameters ------------ #

function merge(sel1::LearnableParameters,
               sel2::LearnableParameters)
    utility = merge(sel1.utility, sel2.utility)
    tree = Dict{Address, Gradients}()
    for k in setdiff(keys(sel2.tree), keys(sel1.tree))
        tree[k] = sel2.tree[k]
    end
    for k in setdiff(keys(sel1.tree), keys(sel2.tree))
        tree[k] = sel1.tree[k]
    end
    for k in intersect(keys(sel1.tree), keys(sel2.tree))
        tree[k] = merge(sel1.tree[k], sel2.tree[k])
    end
    return LearnableParameters(tree, utility)
end

+(a::LearnableParameters, b::LearnableParameters) = merge(a, b)

# ------------ update_parameters links into Flux optimiser APIs ------------ #

function update_parameters(opt, a::LearnableParameters, b::Gradients)
    p_arr = array(a, Float64)
    gs_arr = array(b, Float64)
    update!(opt, p_arr, -gs_arr)
    return selection(a, p_arr)
end

# ------------ Pretty printing utility selections ------------ #

function collect!(par::T, addrs::Vector, chd::Dict, chs::K) where {T <: Tuple, K <: UtilitySelection}
    for (k, v) in chs.utility
        push!(addrs, (par..., k))
        chd[(par..., k)] = v
    end
    for (k, v) in chs.tree
        collect!((par..., k), addrs, chd, v)
    end
end

function collect!(addrs::Vector, chd::Dict, chs::K) where K <: UtilitySelection
    for (k, v) in chs.utility
        push!(addrs, (k, ))
        chd[(k, )] = v
    end
    for (k, v) in chs.tree
        collect!((k, ), addrs, chd, v)
    end
end

import Base.collect
function collect(chs::K) where K <: UtilitySelection
    addrs = []
    chd = Dict()
    collect!(addrs, chd, chs)
    return addrs, chd
end

function Base.display(chs::Gradients; show_values = true)
    println("  __________________________________\n")
    println("             Gradients\n")
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

function Base.display(chs::LearnableParameters; show_values = true)
    println("  __________________________________\n")
    println("             Parameters\n")
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
