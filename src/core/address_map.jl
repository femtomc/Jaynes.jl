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

@inline convert(::Type{Value}, c::Choice) = Value(get_value(c))

@inline length(e::Empty) = 0
@inline length(v::Value) = 1
@inline length(c::Choice) = 1

@inline value_length(e::Empty) = 0
@inline value_length(v::Value{K}) where K = length(get_value(v))
@inline value_length(c::Choice{K}) where K = length(get_value(v))

@inline ndims(e::Empty) = 0
@inline ndims(v::Value{K}) where K = ndims(get_value(v))
@inline ndims(v::Choice{K}) where K = ndims(get_value(v))

@inline projection(c::Choice, tg::Empty) = 0.0
@inline projection(c::Choice, tg::SelectAll) = c.score

@inline filter(fn, l::Leaf) = Empty()
@inline filter(fn, addr, l::Leaf) = Empty()

@inline select(c::Select) = c
@inline select(c::Value) = SelectAll()
@inline select(c::Choice) = SelectAll()

@inline get_score(c::Choice) = c.score

# ------------- Interfaces ------------ #

@inline has_sub(::Leaf, _) = false
@inline has_sub(::Leaf, ::Tuple{}) = false
@inline has_sub(::AddressMap, ::Tuple{}) = false

@inline haskey(::Leaf, _) = false
@inline haskey(::Leaf, ::Tuple{}) = false
@inline haskey(::AddressMap, ::Tuple{}) = false
@inline haskey(am::AddressMap, addr::Tuple{A}) where A <: Address = haskey(am, addr[1])
@inline function haskey(am::AddressMap, addr::Tuple)
    hd, tl = addr[1], addr[2 : end]
    has_sub(am, hd) && haskey(get_sub(am, hd), tl)
end

@inline set_sub!(::Leaf, args...) = error("(set_sub!): trying to set submap of an instance of Leaf type.\nThis normally happens because you've already assigned to this address, or part of the prefix of this address.")
@inline function set_sub!(am::AddressMap{K}, addr::Tuple{}, v::AddressMap{<: K}) where {A <: Address, K} end

@inline get_sub(::Leaf, _) = Empty()
function get_sub(am::AddressMap{K}, addr::Tuple{T})::Union{Empty, AddressMap{K}} where {K, T}
    get_sub(am, addr[1])
end
function get_sub(am::AddressMap{K}, addr::Tuple)::Union{Empty, AddressMap{K}} where K
    get_sub(get_sub(am, addr[1]), addr[2 : end])
end
#Zygote.@adjoint get_sub(a, addr) = get_sub(a, addr), ret_grad -> (ret_grad, nothing)

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
        get_sub(b, k) != sub && return false
    end
    for (k, sub) in shallow_iterator(b)
        get_sub(a, k) != sub && return false
    end
    return true
end

function Base.length(a::AddressMap)
    l = 0
    for (_, sub) in shallow_iterator(a)
        l += length(sub)
    end
    l
end

function value_length(a::AddressMap)
    l = 0
    for (_, sub) in shallow_iterator(a)
        l += value_length(sub)
    end
    l
end

function Base.ndims(a::AddressMap)
    l = 0
    for (_, sub) in shallow_iterator(a)
        l += ndims(sub)
    end
    l
end

@inline Base.merge(am::AddressMap, ::Empty) = deepcopy(am), false
@inline Base.merge(::Empty, am::AddressMap) = deepcopy(am), false
@inline Base.merge(l::Leaf, ::Empty) = deepcopy(l), false
@inline Base.merge(::Empty, l::Leaf) = deepcopy(l), false
@inline Base.merge(c::Choice, v::Value) = deepcopy(c), true
@inline Base.merge(v::Value, c::Choice) = Value(get_value(c)), true
@inline Base.merge!(am::AddressMap, ::Empty) = am, false
@inline Base.merge!(::Empty, am::AddressMap) = am, false
@inline Base.merge!(l::Leaf, ::Empty) = l, false
@inline Base.merge!(::Empty, l::Leaf) = l, false
@inline Base.merge!(c::Choice, v::Value) = v, true
@inline Base.merge!(v::Value, c::Choice) = Value(get_value(c)), true

function collect!(par::T, addrs::Vector, chd::Dict, v::V, meta) where {T <: Tuple, V <: Leaf{Choice}}
    push!(addrs, par)
    chd[par] = get_value(v)
end

function collect!(par::T, addrs::Vector, chd::Dict, v::V, meta) where {T <: Tuple, V <: Leaf{Value}}
    push!(addrs, par)
    chd[par] = get_value(v)
end

function collect!(par::T, addrs::Vector, chd::Dict, v::SelectAll, meta) where T <: Tuple
    push!(addrs, par)
end

function collect(am::M) where M <: AddressMap
    addrs = Any[]
    chd = Dict()
    meta = Dict()
    collect!(addrs, chd, am, meta)
    return addrs, chd, meta
end

function iterate(fn, am::M) where M <: AddressMap
    for (k, v) in shallow_iterator(am)
        fn((k, v))
    end
end

function flatten(am::M) where M <: AddressMap
    addrs, chd, _ = collect(am)
    arr = array(am, Float64)
    addrs, arr, chd
end

function Base.display(am::M; 
                      show_values = true, 
                      show_types = false) where M <: AddressMap
    println(" ___________________________________\n")
    println("             Address Map\n")
    addrs, chd, meta = collect(am)
    if show_values
        for a in addrs
            if haskey(meta, a) && haskey(chd, a)
                println(" $(meta[a]) $(a) = $(chd[a])")
            elseif haskey(chd, a)
                println(" $(a) = $(chd[a])")
            else
                println(" $(a)")
            end
        end
    elseif show_types
        for a in addrs
            if haskey(meta, a) && haskey(chd, a)
                println(" $(meta[a]) $(a) = $(typeof(chd[a]))")
            elseif haskey(chd, a)
                println(" $(a) = $(typeof(chd[a]))")
            else
                println(" $(a)")
            end
        end
    elseif show_types && show_values
        for a in addrs
            if haskey(meta, a) && haskey(chd, a)
                println(" $(meta[a]) $(a) = $(chd[a]) : $(typeof(chd[a]))")
            elseif haskey(chd, a)
                println(" $(a) = $(chd[a]) : $(typeof(chd[a]))")
            else
                println(" $(a)")
            end
        end
    else
        for a in addrs
            println(" $(a)")
        end
    end
    println(" ___________________________________\n")
end

# ------------ includes ------------ #

include("address_maps/dynamic.jl")
include("address_maps/vector.jl")
include("address_maps/conditional.jl")
include("array_compat.jl")
