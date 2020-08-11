# ------------ Address map ------------ #

abstract type AddressMap{K} end

# Leaves.
abstract type Leaf{K} <: AddressMap{K} end

# Leaf interfaces.
isempty(::Leaf) = false
get_ret(::Leaf) = nothing
has_value(::Leaf) = false

struct Empty <: Leaf{Empty} end

isempty(::Empty) = true

struct Value{K} <: Leaf{Value}
    val::K
end

struct ChoiceRecord{K} <: Leaf{Value}
    score::Float64
    val::K
end
get_score(cs::ChoiceRecord) = cs.score

isempty(::K) where K <: Leaf{Value} = false
has_value(::K) where K <: Leaf{Value} = true
get_ret(v::K) where K <: Leaf{Value} = v.val

function collect!(par::T, addrs::Vector{Any}, chd::Dict{Any, Any}, v::V, meta) where {T <: Tuple, V <: Leaf{Value}}
    push!(addrs, par)
    chd[par] = get_ret(v)
end

# Address map interfaces
function fill_array!(val::T, arr::Vector{T}, f_ind::Int) where T
    if length(arr) < f_ind
        resize!(arr, 2 * f_ind)
    end
    arr[f_ind] = val
    1
end

function fill_array!(val::T, arr::Vector{K}, f_ind::Int) where {K, T <: AddressMap}
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

function fill_array!(val::Vector{T}, arr::Vector{T}, f_ind::Int) where T
    if length(arr) < f_ind + length(val)
        resize!(arr, 2 * (f_ind + length(val)))
    end
    arr[f_ind : f_ind + length(val) - 1] = val
    length(val)
end

function array(gs::K, ::Type{T}) where {T, K}
    arr = Vector{T}(undef, 32)
    n = fill_array!(gs, arr, 1)
    resize!(arr, n)
    arr
end

function from_array(::T, arr::Vector{T}, f_ind::Int) where T
    (1, arr[f_ind])
end

function from_array(val::Vector{T}, arr::Vector{T}, f_ind::Int) where T
    n = length(val)
    (n, arr[f_ind : f_ind + n - 1])
end

function from_array(schema::T, arr::Vector{K}, f_ind::Int) where {K, T <: AddressMap}
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

function get_map(schema::K, arr::Vector) where K
    (n, sel) = from_array(schema, arr, 1)
    n != length(arr) && error("Dimension error: length of arr $(length(arr)) must match $n.")
    sel
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

include("address_maps/dynamic.jl")
include("address_maps/anywhere.jl")
include("address_maps/vector.jl")
