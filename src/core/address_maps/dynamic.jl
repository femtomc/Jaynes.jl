# ------------ Dynamic map ------------ #

struct DynamicMap{K} <: AddressMap{K}
    tree::Dict{Any, AddressMap{<:K}}
    DynamicMap{K}() where K = new{K}(Dict{Any, AddressMap{K}}())
    DynamicMap{K}(tree::Dict{Any, AddressMap{K}}) where K = new{K}(tree)
end
DynamicMap(tree::Dict{Any, AddressMap{K}}) where K = DynamicMap{K}(tree)
Zygote.@adjoint DynamicMap(tree) = DynamicMap(tree), ret_grad -> (nothing, )

@inline shallow_iterator(dm::DynamicMap) = dm.tree

@inline function get(dm::DynamicMap{Value}, addr, fallback)
    haskey(dm, addr) || return fallback
    return getindex(dm, addr)
end

@inline function get_sub(dm::DynamicMap, addr::A) where A <: Address
    haskey(dm.tree, addr) && return getindex(dm.tree, addr)
    Empty()
end
@inline get_sub(dm::DynamicMap, addr::Tuple{}) = Empty()
@inline getindex(dm::DynamicMap, addrs...) = get_value(get_sub(dm, addrs))

@inline Base.isempty(dm::DynamicMap) = isempty(dm.tree)

# This is a fallback for subtypes of DynamicMap.
@inline haskey(dm::DynamicMap, addr::A) where A <: Address = haskey(dm.tree, addr) && has_value(get_sub(dm, addr))

# This is a fallback for subtypes of DynamicMap.
@inline has_sub(dm::DynamicMap, addr::A) where A <: Address = haskey(dm.tree, addr)

function set_sub!(dm::DynamicMap{K}, addr::A, v::AddressMap{<:K}) where {K, A <: Address}
    delete!(dm.tree, addr)
    if !isempty(v)
        dm.tree[addr] = v
    end
end
@inline set_sub!(dm::DynamicMap{K}, addr::Tuple{A}, v::AddressMap{<: K}) where {A <: Address, K} = set_sub!(dm, addr[1], v)
function set_sub!(dm::DynamicMap{K}, addr::Tuple, v::AddressMap{<:K}) where K
    hd, tl = addr[1], addr[2 : end]
    if !haskey(dm.tree, hd)
        dm.tree[hd] = DynamicMap{K}()
    end
    set_sub!(dm.tree[hd], tl, v)
end

function merge(sel1::DynamicMap{K},
               sel2::DynamicMap{K}) where K
    tree = Dict{Any, AddressMap{K}}()
    for k in setdiff(keys(sel2.tree), keys(sel1.tree))
        tree[k] = sel2.tree[k]
    end
    for k in setdiff(keys(sel1.tree), keys(sel2.tree))
        tree[k] = sel1.tree[k]
    end
    inter = intersect(keys(sel1.tree), keys(sel2.tree))
    for k in inter
        tree[k] = merge(sel1.tree[k], sel2.tree[k])
    end
    return DynamicMap(tree), !isempty(inter)
end
function merge(sel1::DynamicMap{T},
               sel2::DynamicMap{K}) where {T, K}
    tree = Dict{Any, AddressMap{K}}()
    for k in setdiff(keys(sel2.tree), keys(sel1.tree))
        tree[k] = sel2.tree[k]
    end
    for k in setdiff(keys(sel1.tree), keys(sel2.tree))
        tree[k] = convert(K, sel1.tree[k])
    end
    inter = intersect(keys(sel1.tree), keys(sel2.tree))
    for k in inter
        tree[k] = merge(sel1.tree[k], sel2.tree[k])
    end
    return DynamicMap(tree), !isempty(inter)
end
merge(dm::DynamicMap, ::Empty) = Empty(), false
merge(::Empty, dm::DynamicMap) = deepcopy(dm), false

function merge!(sel1::DynamicMap{K},
                sel2::DynamicMap{K}) where K
    inter = intersect(keys(sel1.tree), keys(sel2.tree))
    for k in setdiff(keys(sel2.tree), keys(sel1.tree))
        set_sub!(sel1, k, get_sub(sel2, k))
    end
    for k in inter
        set_sub!(sel1, k, get_sub(sel2, k))
    end
    !isempty(inter)
end
merge!(dm::DynamicMap, ::Empty) = Empty(), false
merge!(::Empty, dm::DynamicMap) = dm, false
Zygote.@adjoint merge!(a, b) = merge!(a, b), s_grad -> (nothing, nothing)

function collect!(par::T, addrs::Vector{Any}, chd::Dict{Any, Any}, tr::D, meta) where {T <: Tuple, D <: DynamicMap}
    iterate(tr) do (k, v)
        collect!((par..., k), addrs, chd, v, meta)
    end
end
function collect!(addrs::Vector{Any}, chd::Dict{Any, Any}, tr::D, meta) where D <: DynamicMap
    iterate(tr) do (k, v)
        collect!((k, ), addrs, chd, v, meta)
    end
end

@inline target() = DynamicMap{Value}()
function target(v::Vector{Pair{T, K}}) where {T <: Tuple, K}
    tg = DynamicMap{Value}()
    for (k, v) in v
        set_sub!(tg, k, Value(v))
    end
    tg
end

# Filter.
function filter(fn, dm::DynamicMap{K}) where K
    new = DynamicMap{K}()
    for (k, v) in shallow_iterator(dm)
        if fn((k, ))
            set_sub!(new, k, v)
        else
            ret = filter(fn, (k, ), v)
            !isempty(ret) && set_sub!(new, k, filter(fn, (k, ), v))
        end
    end
    new
end

function filter(fn, par, dm::DynamicMap{K}) where K
    new = DynamicMap{K}()
    for (k, v) in shallow_iterator(dm)
        if fn((par..., k))
            set_sub!(new, k, v)
        else
            ret = filter(fn, (par..., k), v)
            !isempty(ret) && set_sub!(new, k, filter(fn, (par..., k), v))
        end
    end
    new
end

# Select.
function select(dm::DynamicMap{K}) where K
    new = DynamicTarget()
    for (k, v) in shallow_iterator(dm)
        set_sub!(new, k, select(v))
    end
    new
end
