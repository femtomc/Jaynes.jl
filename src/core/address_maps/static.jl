struct StaticMap{T, K, B} <: AddressMap{K}
    tree::NamedTuple{T, <: K}
    isempty::Val{B}
    function StaticMap{K}() where K
        new{(), K, true}(NamedTuple(), Val(true))
    end
    function StaticMap{K}(tree::Dict{Symbol, AddressMap{K}}) where K
        nt = (; tree...)
        new{keys(nt), K, Val{isempty(tree)}}(nt, Val(isempty(true)))
    end
end
StaticMap(tree::Dict{Symbol, AddressMap{K}}) where K = StaticMap{K}(tree)

@inline shallow_iterator(sm::StaticMap) = dm.tree

@inline function get(sm::StaticMap{T, Value}, addr, fallback) where T
    haskey(sm, addr) || return fallback
    return getindex(sm, addr)
end

@inline function get_sub(sm::StaticMap, addr::A) where A <: Address
    haskey(sm.tree, addr) && return getindex(sm.tree, addr)
    Empty()
end
@inline get_sub(sm::StaticMap, addr::Tuple{}) = Empty()

@inline getindex(sm::StaticMap, addrs...) = get_value(get_sub(sm, addrs))

@inline Base.isempty(sm::StaticMap{T, K, Val{false}}) where {T, K, B} = false
@inline Base.isempty(sm::StaticMap{T, K, Val{true}}) where {T, K, B} = true

@inline haskey(sm::StaticMap, addr::A) where A <: Address = haskey(sm.tree, addr) && has_value(get_sub(sm, addr))

@inline has_sub(sm::StaticMap, addr::A) where A <: Address = haskey(dm.tree, addr)

function set_sub(sm::StaticMap{T, K, B}, addr::A, v::AddressMap{<: K}) where {T, K, B, A <: Address}
    d = Dict{Symbol, AddressMap{K}}(map(keys(sm.tree)) do k
                 k => getindex(sm, k)
             end)
    d[addr] = v
    StaticMap(d)
end
@inline set_sub(sm::StaticMap{T, K, B}, addr::Tuple{A}, v::AddressMap{<: K}) where {T, K, B, A <: Address} = set_sub(sm, addr[1], v)
function set_sub(sm::StaticMap{T, K, B}, addr::Tuple, v::AddressMap{<: K}) where {T, K, B}
    hd, tl = addr[1], addr[2 : end]
    d = Dict{Symbol, AddressMap{K}}()
    if !haskey(dm.tree, hd)
        d[hd] = StaticMap{K}()
    end
    set_sub(d[hd], tl, v)
    StaticMap(d)
end

function merge(sel1::StaticMap{T, K, B1},
               sel2::StaticMap{L, K, B2}) where {T, K, B1, L, B2}
    tree = Dict{Any, AddressMap{K}}()
    inter = intersect(keys(sel1.tree), keys(sel2.tree))
    check = false
    for k in setdiff(keys(sel2.tree), keys(sel1.tree))
        tree[k] = sel2.tree[k]
    end
    for k in setdiff(keys(sel1.tree), keys(sel2.tree))
        tree[k] = sel1.tree[k]
    end
    for k in inter
        tree[k], b = merge(sel1.tree[k], sel2.tree[k])
        check = check || b
    end
    return StaticMap((; tree...)), !isempty(inter) || check
end

merge(sm::StaticMap, ::Empty) = deepcopy(sm), false
merge(::Empty, sm::StaticMap) = deepcopy(sm), false

function collect!(par::T, addrs::Vector{Any}, chd::Dict{Any, Any}, tr::S, meta) where {T <: Tuple, S <: StaticMap}
    iterate(tr) do (k, v)
        collect!((par..., k), addrs, chd, v, meta)
    end
end
function collect!(addrs::Vector{Any}, chd::Dict{Any, Any}, tr::S, meta) where S <: StaticMap
    iterate(tr) do (k, v)
        collect!((k, ), addrs, chd, v, meta)
    end
end

function static(v::Vector{Pair{T, K}}) where {T <: Tuple, K}
    sm = StaticMap{Any}()
    for (k, v) in v
        sm = set_sub(sm, k, v)
    end
    sm
end

function static(v::Pair{T, K}) where {T <: Tuple, K}
    set_sub(StaticMap{Value}(), v[1], Value(v[2]))
end
