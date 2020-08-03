# TODO: replace with Mjolnir version.

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

# Register diff propagation with Mjolnir
#struct DiffPrimitives end
#
#@abstract DiffPrimitives function *(d1::Partial{<:Diffed{V, NoChange}}, d2::Partial{<:Diffed{V, NoChange}}) where V
#    Diffed(d1.value.val * d2.value.val, NoChange())
#end
#@abstract DiffPrimitives function +(d1::Partial{<:Diffed{V, NoChange}}, d2::Partial{<:Diffed{V, NoChange}}) where V
#    Diffed(d1.value.val + d2.value.val, NoChange())
#end
#@abstract DiffPrimitives function -(d1::Partial{<:Diffed{V, NoChange}}, d2::Partial{<:Diffed{V, NoChange}}) where V
#    Diffed(d1.value.val - d2.value.val, NoChange())
#end
#@abstract DiffPrimitives function /(d1::Partial{<:Diffed{V, NoChange}}, d2::Partial{<:Diffed{V, NoChange}}) where V
#    Diffed(d1.value.val / d2.value.val, NoChange())
#end
