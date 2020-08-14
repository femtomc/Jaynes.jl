# ------------ Dynamic map ------------ #

struct DynamicMap{K} <: AddressMap{K}
    tree::Dict{Any, AddressMap{<:K}}
    DynamicMap{K}() where K = new{K}(Dict{Any, AddressMap{K}}())
    DynamicMap{K}(tree::Dict{Any, AddressMap{<:K}}) where K = new{K}(tree)
end
DynamicMap(tree::Dict{Any, AddressMap{K}}) where K = DynamicMap{K}(tree)
Zygote.@adjoint DynamicMap(tree) = DynamicMap(tree), ret_grad -> (nothing, )

@inline shallow_iterator(dm::DynamicMap) = dm.tree

@inline get_sub(dm::DynamicMap, addr) = get(dm.tree, addr, Empty())
@inline get_sub(dm::DynamicMap, addr::Tuple{}) = Empty()
@inline getindex(dm::DynamicMap, addr) = get_value(get_sub(dm, addr))

@inline Base.isempty(dm::DynamicMap) = isempty(dm.tree)

function haskey(dm::DynamicMap, addr)
    haskey(dm.tree, addr) && !isempty(dm.tree[addr])
end

function set_sub!(dm::DynamicMap{K}, addr, v::AddressMap{<:K}) where K
    delete!(dm.tree, addr)
    if !isempty(v)
        dm.tree[addr] = v
    end
end
function set_sub!(dm::DynamicMap{K}, addr::Tuple{}, v::AddressMap{<:K}) where {K, T} end
function set_sub!(dm::DynamicMap{K}, addr::Tuple{T}, v::AddressMap{<:K}) where {K, T}
    set_sub!(dm, addr[1], v)
end
function set_sub!(dm::DynamicMap{K}, addr::Tuple, v::AddressMap{<:K}) where K
    hd, tl = addr
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
    for k in intersect(keys(sel1.tree), keys(sel2.tree))
        tree[k] = merge(sel1.tree[k], sel2.tree[k])
    end
    return DynamicMap(tree)
end
merge(dm::DynamicMap, ::Empty) = deepcopy(dm)
merge(::Empty, dm::DynamicMap) = deepcopy(dm)
Zygote.@adjoint merge(a, b) = merge(a, b), s_grad -> (nothing, nothing)
+(a::DynamicMap{K}, b::DynamicMap{K}) where K = merge(a, b)

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
