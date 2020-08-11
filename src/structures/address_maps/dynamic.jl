# ------------ Dynamic map ------------ #

struct DynamicMap{K} <: AddressMap{K}
    tree::Dict{Any, AddressMap{<:K}}
    DynamicMap{K}() where K = new{K}(Dict{Any, AddressMap{K}}())
    DynamicMap{K}(tree::Dict{Any, AddressMap{<:K}}) where K = new{K}(tree)
end

DynamicMap(tree::Dict{Any, AddressMap{K}}) where K = DynamicMap{K}(tree)
Zygote.@adjoint DynamicMap(tree) = DynamicMap(tree), retgrad -> (nothing, )

# Dynamic map interfaces
push!(dm::DynamicMap{K}, addr, v::AddressMap{<:K}) where K = dm.tree[addr] = v

function haskey(dm::DynamicMap, addr)
    haskey(dm.tree, addr) && !isempty(dm.tree[addr])
end

get_ret(dm::DynamicMap) = dm

function get_submap(dm::DynamicMap, addr)
    haskey(dm, addr) && return dm.tree[addr]
    Empty()
end

has_value(dm::DynamicMap, addr) = has_value(get_submap(dm, addr))
has_value(dm::DynamicMap) = false

function isempty(dm::DynamicMap)
    for (k, v) in dm.tree
        !isempty(v) && return false
    end
    return true
end

function getindex(dm::DynamicMap, addr)
    val = dm.tree[addr]
    isempty(val) ? Empty() : get_ret(val)
end

function get_leaf(dm::DynamicMap, addr)
    return dm.tree[addr]
end

function setindex!(dm::DynamicMap{K}, val::AddressMap{<:K}, addr) where K
    dm.tree[addr] = val
end

function set_submap!(dm::DynamicMap{K}, addr, val::AddressMap{<: K}) where K
    setindex!(dm, val, addr)
end

function get(dm::DynamicMap{K}, addr, val) where K
    haskey(dm, addr) && return getindex(dm, addr)
    return val
end

function get_iter(dm::DynamicMap)
    return (
            (k, v) for (k, v) in dm.tree
           )
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
    for k in intersect(keys(sel1.tree), keys(sel2.tree))
        tree[k] = merge(sel1.tree[k], sel2.tree[k])
    end
    return DynamicMap(tree)
end

Zygote.@adjoint merge(a, b) = merge(a, b), s_grad -> (nothing, nothing)

+(a::DynamicMap{K}, b::DynamicMap{K}) where K = merge(a, b)

function iterate(fn, tr::D) where D <: DynamicMap
    for (k, v) in tr.tree
        fn((k, v))
    end
end

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

# ------------ Dynamic value map ------------ #

function dynamic(sel::Vector{Pair{T, K}}) where {T <: Tuple, K}
    top = Trace()
    for (k, v) in sel
        push!(top, k, Value(v))
    end
    top
end

function push!(dm::M, addr::Tuple{T}, v::AddressMap{<:K}) where {M <: DynamicMap{Value}, T, K <: Value}
    push!(dm, addr[1], v)
end

function push!(dm::M, addr::T, v::AddressMap{<:K}) where {M <: DynamicMap{Value}, T <: Tuple, K <: Value}
    sub = DynamicMap{Value}()
    push!(sub, addr[2 : end], v)
    dm[addr[1]] = sub
end
