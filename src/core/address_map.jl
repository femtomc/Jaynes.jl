# ------------ Address map ------------ #

abstract type AddressMap{K} end

# Leaves.
abstract type Leaf{K} <: AddressMap{K} end

struct Empty <: Leaf{Empty} end

abstract type Select <: Leaf{Select} end
struct SelectAll <: Select end
struct Special <: Leaf{Special} end

struct Value{K} <: Leaf{Value}
    val::K
end

struct Choice{K} <: Leaf{Choice}
    score::Float64
    val::K
end

@inline projection(c::Choice, tg::Empty) = 0.0
@inline projection(c::Choice, tg::SelectAll) = c.score

@inline get_score(c::Choice) = c.score

# ------------- Interfaces ------------ #

@inline set_sub!(::Leaf, args...) = error("(set_sub!): trying to set submap of an instance of Leaf type.\nThis normally happens because you've already assigned to this address, or part of the prefix of this address.")
@inline get_sub(::Leaf, _) = Empty()
function get_sub(::AddressMap{K}, addr)::AddressMap{K} where K end
function get_sub(::AddressMap{K}, addr::Tuple{T})::AddressMap{K} where {K, T}
    get_sub(am, addr[1])
end
function get_sub(am::AddressMap{K}, addr::Tuple)::AddressMap{K} where K
    get_sub(get_sub(am, addr[1]), addr[2 : end])
end

function shallow_iterator(::AddressMap{K}) where K end
@inline shallow_iterator(::Leaf) = ()

@inline isempty(::Empty) = true
@inline isempty(::Value) = false
@inline isempty(::Choice) = false
@inline isempty(::SelectAll) = false
@inline isempty(am::AddressMap) = all(shallow_iterator(am)) do (_, v)
    iempty(v)
end

@inline has_value(am::AddressMap, addr) = has_value(get_sub(am, addr))
@inline has_value(am::AddressMap) = false

@inline get_value(v::Value) = v.val
@inline get_value(v::Choice) = v.val
@inline has_value(v::Value) = true
@inline has_value(v::Choice) = true

@inline Base.:(==)(a::Value, b::Value) = a.val == b.val

# TODO: inefficient.
function Base.:(==)(a::AddressMap, b::AddressMap)
    for (k, sub) in shallow_iterator(a)
        get_sub(b, addr) != sub && return false
    end
    for (k, sub) in shallow_iterator(b)
        get_sub(a, addr) != sub && return false
    end
    return true
end

@inline Base.merge(am::AddressMap, ::Empty) = deepcopy(am)
@inline Base.merge(::Empty, am::AddressMap) = deepcopy(am)
@inline Base.merge(l::Leaf, ::Empty) = deepcopy(l)
@inline Base.merge(::Empty, l::Leaf) = deepcopy(l)
@inline Base.merge!(am::AddressMap, ::Empty) = am
@inline Base.merge!(::Empty, am::AddressMap) = am
@inline Base.merge!(l::Leaf, ::Empty) = l
@inline Base.merge!(::Empty, l::Leaf) = l

function collect!(par::T, addrs::Vector{Any}, chd::Dict{Any, Any}, v::V, meta) where {T <: Tuple, V <: Leaf{Value}}
    push!(addrs, par)
    chd[par] = get_ret(v)
end

function collect(tr::M) where M <: AddressMap
    addrs = Any[]
    chd = Dict{Any, Any}()
    meta = Dict()
    collect!(addrs, chd, tr, meta)
    return addrs, chd, meta
end

function iterate(fn, am::M) where M <: AddressMap
    for (k, v) in shallow_iterator(am)
        fn((k, v))
    end
end

function Base.display(tr::D; 
                      show_values = true, 
                      show_types = false) where D <: AddressMap
    println(" ___________________________________\n")
    println("             Address Map\n")
    addrs, chd, meta = collect(tr)
    if show_values
        for a in addrs
            if haskey(meta, a)
                println(" $(meta[a]) $(a) = $(chd[a])")
            else
                println(" $(a) = $(chd[a])")
            end
        end
    elseif show_types
        for a in addrs
            println(" $(a) = $(typeof(chd[a]))")
        end
    elseif show_types && show_values
        for a in addrs
            println(" $(a) = $(chd[a]) : $(typeof(chd[a]))")
        end
    else
        for a in addrs
            println(" $(a)")
        end
    end
    println(" ___________________________________\n")
end

# ------------ includes ------------ #

include("array_compat.jl")
include("address_maps/dynamic.jl")
include("address_maps/anywhere.jl")
include("address_maps/vector.jl")
