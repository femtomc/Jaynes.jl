# ------------ Diff system ------------ #

abstract type Diff end

struct UndefinedChange <: Diff end

struct NoChange <: Diff end

struct SetDiff{V} <: Diff
    added::Set{V}
    deleted::Set{V}
end

struct DictDiff{K, V} <: Diff
    added::AbstractDict{K, V}
    deleted::AbstractSet{K}
    updated::AbstractDict{K, Diff}
end

struct VectorDiff <: Diff
    new_length::Int
    prev_length::Int
    updated::Dict{Int,Diff} 
    VectorDiff(nl::Int, pl::Int, updated::Dict{Int, Diff}) = new(nl, pl, updated)
end

struct IntDiff <: Diff
    difference::Int # new - old
end

struct Diffed{V, DV}
    val::V
    diff::DV
end

# Define the algebra for propagation of diffs.
propagate(::NoChange, ::NoChange) = NoChange()
propagate(::UndefinedChange, ::NoChange) = UndefinedChange()
propagate(::NoChange, ::UndefinedChange) = UndefinedChange()
propagate(::UndefinedChange, ::UndefinedChange) = UndefinedChange()
propagate(a::Type{NoChange}, v::Type{NoChange}) = NoChange()
propagate(a::Type{NoChange}, v::Type{UndefinedChange}) = UndefinedChange()
propagate(a::Type{UndefinedChange}, v::Type{NoChange}) = UndefinedChange()
propagate(a::Type{UndefinedChange}, v::Type{UndefinedChange}) = UndefinedChange()
propagate(a::Type{K}, b::T) where {K, T} = propagate(K, T)

struct DiffPrimitives end

#include("lib/numeric.jl")
#include("lib/distributions.jl")
