# ------------ Solo map ------------ #

mutable struct SoloMap{K} <: AddressMap{K}
    pointer::AddressMap{<: K}
    SoloMap{K}() where K = new{K}()
    SoloMap(pointer::K) where K = new{K}(pointer)
end

@inline shallow_iterator(sm::SoloMap{K}) where K = NVector((:pointer, sm.pointer))

@inline function get(sm::SoloMap{Value}, addr, fallback)
    hasfield(sm, addr) || return fallback
    return getfield(sm, addr)
end

@inline function get_sub(sm::SoloMap, addr::A) where A <: Address
    hasfield(sm, addr) && return getfield(sm, addr)
    Empty()
end
@inline get_sub(sm::SoloMap, addr::Tuple{}) = Empty()

@inline getindex(sm::SoloMap, addrs...) = get_value(get_sub(sm, addrs))

@inline Base.isempty(sm::SoloMap) = isempty(getfield(sm, :pointer))

@inline haskey(sm::SoloMap, addr::A) where A <: Address = hasfield(sm, addr) && has_value(get_sub(sm, addr))

@inline has_sub(sm::SoloMap, addr::A) where A <: Address = hasfield(sm, addr)

function set_sub!(sm::SoloMap{K}, addr::A, v::AddressMap{<: K}) where {K, A <: Address}
    hasfield(sm, addr) || return
    !isempty(v) && setfield!(sm, addr, v)
end
function set_sub!(sm::SoloMap{K}, addr::A, v::V) where {K, V, A <: Address}
    hasfield(sm, addr) || return
    !isempty(v) && setfield!(sm, addr, convert(K, v))
end

function merge(sel1::SoloMap{K}, sel2::SoloMap{K}) where K
    new = SoloMap{K}()
    setfield!(new, :pointer, sel2.pointer)
    new, true
end
merge(sm::SoloMap, ::Empty) = SoloMap(sm.pointer), false
merge(::Empty, sm::SoloMap) = SoloMap(sm.pointer), false

function merge!(sel1::SoloMap{K}, sel2::SoloMap{K}) where K
    setfield!(sel1, :pointer, sel2.pointer)
    true
end
merge!(sm::SoloMap, ::Empty) = sm, false
merge!(::Empty, sm::SoloMap) = sm, false

function collect!(par::T, addrs::Vector{Any}, chd::Dict{Any, Any}, tr::S, meta) where {T <: Tuple, S <: SoloMap}
    iterate(tr) do (k, v)
        collect!((par..., k), addrs, chd, v, meta)
    end
end
function collect!(addrs::Vector{Any}, chd::Dict{Any, Any}, tr::S, meta) where S <: SoloMap
    iterate(tr) do (k, v)
        collect!((k, ), addrs, chd, v, meta)
    end
end

# Select.
function select(sm::SoloMap{K}) where K
    new = DynamicTarget()
    for (k, v) in shallow_iterator(sm)
        set_sub!(new, k, select(v))
    end
    new
end
