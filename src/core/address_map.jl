# ------------ Address map ------------ #

abstract type AddressMap{K} end

# Leaves.
abstract type Leaf{K} <: AddressMap{K} end

struct Empty <: Leaf{Empty} end

struct Value{K} <: Leaf{Value}
    val::K
end

@inline isempty(::Empty) = true
@inline isempty(::Value) = false
@inline isempty(am::AddressMap) = all(shallow_iterator(am)) do (_, v)
    iempty(v)
end

@inline get_sub(::Leaf, _) = Empty()

@inline Base.:(==)(a::Value, b::Value) = a.val == b.val

@inline get_value(v::Value) = v.val
@inline has_value(::Value) = true

@inline shallow_iterator(::Leaf) = ()

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

include("trie.jl")
include("array_compat.jl")
include("address_maps/dynamic.jl")
include("address_maps/anywhere.jl")
include("address_maps/vector.jl")
