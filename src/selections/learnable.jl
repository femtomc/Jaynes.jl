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

getindex(ps::Gradients, addr::T) where T <: Address = getindex(ps.utility, addr)
getindex(ps::Gradients, addr::Tuple{}) = error("Gradients (getindex): empty tuple as index.")
getindex(ps::Gradients, addr::Tuple{T}) where T <: Address = getindex(ps, addr[1])
function getindex(ps::Gradients, addr::T) where T <: Tuple
    has_sub(ps, addr[1]) && return getindex(get_sub(ps, addr[1]), addr[2 : end])
    error("Gradients (getindex): invalid index at $addr.")
end
function getindex(ps::Gradients, addrs...)
    has_top(ps, addrs) && return get_top(ps, addrs)
    has_sub(ps, addrs) && return get_sub(ps, addrs)
    error("Gradients (getindex): invalid index at $addr.")
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

# ------------ Learnable by address ------------ #

struct LearnableByAddress <: Parameters
    tree::Dict{Address, LearnableByAddress}
    utility::Dict{Address, Any}
    LearnableByAddress() = new(Dict{Address, LearnableByAddress}(), Dict{Address, Any}())
end

has_sub(ps::LearnableByAddress, addr::T) where T <: Address = haskey(ps.tree, addr)
function has_sub(ps::LearnableByAddress, addr::T) where T <: Tuple
    isempty(addr) && return false
    length(addr) == 1 && return has_sub(ps, addr[1])
    has_sub(ps, addr[1]) && has_sub(get_sub(ps, addr[1]), addr[2 : end])
end

function get_sub(ps::LearnableByAddress, addr::Tuple{})
    Parameters()
end
function get_sub(ps::LearnableByAddress, addr::T) where T <: Address
    has_sub(ps, addr) || return Parameters()
    getindex(ps.tree, addr)
end
get_sub(ps::LearnableByAddress, addr::Tuple{T}) where T <: Address = get_sub(ps, addr[1])
function get_sub(ps::LearnableByAddress, addr::T) where T <: Tuple
    has_sub(ps, addr[1]) || return Parameters()
    get_sub(ps.tree[addr[1]], addr[2 : end])
end

set_sub!(ps::LearnableByAddress, addr::T, sub::LearnableByAddress) where T <: Address = ps.tree[addr] = sub
function set_sub!(ps::LearnableByAddress, addr::T, sub::LearnableByAddress) where T <: Tuple
    isempty(addr) && return
    length(addr) == 1 && set_sub!(ps, addr[1], sub)
    has_sub(ps, addr[1]) && set_sub!(get_sub(ps, addr[1]), addr[2 : end], sub)
end

has_top(ps::LearnableByAddress, addr::T) where T <: Address = haskey(ps.utility, addr)
function has_top(ps::LearnableByAddress, addr::T) where T <: Tuple
    isempty(addr) && return false
    length(addr) == 1 && return has_top(ps, addr[1])
    has_sub(ps, addr[1]) && return has_top(get_sub(ps, addr[1]), addr[2 : end])
end

get_top(ps::LearnableByAddress, addr::T) where T <: Address = getindex(ps.utility, addr)
function get_top(ps::LearnableByAddress, addr::T) where T <: Tuple
    isempty(addr) && error("LearnableByAddress (get_top): invalid index at $addr.")
    length(addr) == 1 && return get_top(ps, addr[1])
    get_top(ps.tree[addr[1]], addr[2 : end])
end

getindex(ps::LearnableByAddress, addrs...) = get_top(ps, addrs)

# ------------ Builders ------------ #

function push!(ps::LearnableByAddress, addr::T, val) where T <: Address
    ps.utility[addr] = val
end

function push!(ps::LearnableByAddress, addr::T, val) where T <: Tuple
    isempty(addr) && return
    length(addr) == 1 && push!(ps, addr[1], val)
    hd = addr[1]
    tl = addr[2 : end]
    if has_sub(ps, hd)
        push!(get_sub(ps, hd), tl, val)
    else
        sub = LearnableByAddress()
        push!(sub, tl, val)
        ps.tree[hd] = sub
    end
end

function learnables(arr::Array{Pair{T, K}}) where {T <: Tuple, K}
    top = LearnableByAddress()
    map(arr) do (k, v)
        push!(top, k, v)
    end
    return top
end

function learnables(p::Pair{T, K}) where {T <: Symbol, K <: Parameters}
    top = LearnableByAddress()
    set_sub!(top, p[1], p[2])
    return top
end

# ------------ Adjoint ------------ #

Zygote.@adjoint LearnableByAddress(tree, utility) = LearnableByAddress(tree, utility), s_grad -> (nothing, nothing)

# ------------ Merging two sets of parameters ------------ #

function merge(sel1::LearnableByAddress,
               sel2::LearnableByAddress)
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
    return LearnableByAddress(tree, utility)
end

+(a::LearnableByAddress, b::LearnableByAddress) = merge(a, b)

# ------------ update_learnables links into Flux optimiser APIs ------------ #

function update_learnables(opt, a::LearnableByAddress, b::Gradients)
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

function Base.display(chs::LearnableByAddress; show_values = true)
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

# ------------ Learnable anywhere ------------ #

struct LearnableAnywhere <: Parameters
    utility::Dict{Address, Any}
    LearnableAnywhere(obs::Vector{Tuple{T, K}}) where {T <: Any, K} = new(Dict{Address, Any}(obs))
    LearnableAnywhere(obs::Tuple{T, K}...) where {T <: Any, K} = new(Dict{Address, Any}(collect(obs)))
end

has_sub(ps::LearnableAnywhere, addr) = true
get_sub(ps::LearnableAnywhere, addr) = ps

has_top(ps::LearnableAnywhere, addr::T) where T <: Address = haskey(ps.utility, addr)
has_top(ps::LearnableAnywhere, addr::Tuple{}) = false
has_top(ps::LearnableAnywhere, addr::Tuple{T}) where T <: Address = has_top(ps, addr[1])
function has_top(ps::LearnableAnywhere, addr::T) where T <: Tuple
    has_sub(ps, addr[1]) && return has_top(get_sub(ps, addr[1]), addr[2 : end])
end

get_top(ps::LearnableAnywhere, addr::T) where T <: Address = getindex(ps.utility, addr)
get_top(ps::LearnableAnywhere, addr::Tuple{}) = Parameters()
get_top(ps::LearnableAnywhere, addr::Tuple{T}) where T <: Address = get_top(ps, addr[1])
get_top(ps::LearnableAnywhere, addr::T) where T <: Tuple = get_top(ps, addr[end])
getindex(ps::LearnableAnywhere, addrs...) = get_top(ps, addrs)
