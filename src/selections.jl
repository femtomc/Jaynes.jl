import Base: haskey, getindex, push!, merge!, union, isempty, merge
import Base.collect
import Base.filter
import Base: +

# ------------ Selection ------------ #

# Abstract type for a sort of query language for addresses within a particular method body.
abstract type Selection end

# ------------ Generic conversion to and from arrays ------------ #

# Subtypes of Selection should implement their own fill_array! and from_array.
# Right now, these methods are really only used for gradient-based learning.

function array(gs::Selection, ::Type{T}) where T
    arr = Vector{T}(undef, 32)
    n = fill_array!(gs, arr, 1)
    resize!(arr, n)
    arr
end

function fill_array!(val::T, arr::Vector{T}, f_ind::Int) where T
    if length(arr) < f_ind
        resize!(arr, 2 * f_ind)
    end
    arr[f_ind] = val
    1
end

function fill_array!(val::Vector{T}, arr::Vector{T}, f_ind::Int) where T
    if length(arr) < f_ind + length(val)
        resize!(arr, 2 * (f_ind + length(val)))
    end
    arr[f_ind : f_ind + length(val) - 1] = val
    length(val)
end

function selection(schema::Selection, arr::Vector)
    (n, sel) = from_array(schema, arr, 1)
    n != length(arr) && error("Dimension error: length of arr $(length(arr)) must match $n.")
    sel
end

function from_array(::T, arr::Vector{T}, f_ind::Int) where T
    (1, arr[f_ind])
end

function from_array(val::Vector{T}, arr::Vector{T}, f_ind::Int) where T
    n = length(val)
    (n, arr[f_ind : f_ind + n - 1])
end

# ------------ Visitor ------------ #

include("selections/visitor.jl")

# ------------ Constrained and unconstrained selections ------------ #

abstract type ConstrainedSelection end
abstract type UnconstrainedSelection end
abstract type SelectQuery <: Selection end
abstract type ConstrainedSelectQuery <: SelectQuery end
abstract type UnconstrainedSelectQuery <: SelectQuery end

# Note: there are selections and "query" structs which can be used into selections.

include("selections/empty.jl")
include("selections/by_address.jl")
include("selections/anywhere.jl")
include("selections/hierarchical.jl")
include("selections/union.jl")

# ------------ Convenience constructor ------------ #

selection() = ConstrainedEmptySelection()

function selection(a::Vector{Tuple{K, T}}; anywhere = false) where {T, K}
    anywhere && return ConstrainedAnywhereSelection(a)
    return ConstrainedHierarchicalSelection(a)
end

function selection(a::Vector{Tuple{Int, T}}; anywhere = false) where T
    return ConstrainedHierarchicalSelection(a)
end

function selection(a::Vector{K}) where K <: Union{Symbol, Pair}
    return UnconstrainedHierarchicalSelection(a)
end

function selection(a::Tuple{K, T}...) where {T, K <: Union{Symbol, Pair}}
    observations = Vector{Tuple{K, T}}(collect(a))
    return ConstrainedHierarchicalSelection(observations)
end

function selection(p::Pair{K, C}) where {K <: Address, C <: ConstrainedSelection}
    top = ConstrainedHierarchicalSelection()
    set_sub!(top, p[1], p[2])
    return top
end

function selection(a::Address...)
    observations = Vector{Address}(collect(a))
    return UnconstrainedHierarchicalSelection(observations)
end

# ------------ Set operations on selections ------------ #

union(a::ConstrainedSelection...) = ConstrainedUnionSelection([a...])
function intersection(a::ConstrainedSelection...) end

# ------------ Documentation ------------ #

