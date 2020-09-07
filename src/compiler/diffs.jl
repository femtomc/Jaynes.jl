# ------------ Diff system ------------ #

abstract type Diff end

abstract type StaticDiff <: Diff end
struct UndefinedChange <: StaticDiff end
struct NoChange <: StaticDiff end

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

unwrap(::Mjolnir.Node{T}) where T = T

# Define the algebra for propagation of diffs.
@inline propagate(a...) = any(i -> unwrap(i) == UndefinedChange, a) ? UndefinedChange : NoChange

struct DiffPrimitives end

include("lib/numeric.jl")
include("lib/distributions.jl")
