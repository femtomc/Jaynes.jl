import Base.rand
rand(addr::T, d::Distribution{K}) where {K, T <: Union{Symbol, Pair{Symbol, Int64}}} = rand(d)
learnable(addr::T, lit::K) where {T <: Union{Symbol, Pair{Symbol, Int64}}, 
                                  K <: Union{Number, AbstractArray}} = lit

abstract type Trace end
abstract type RecordSite end

struct ChoiceSite{T} <: RecordSite
    val::T
    score::Float64
end

mutable struct CallSite{T <: Trace, J, K} <: RecordSite
    trace::T
    fn::Function
    args::J
    ret::K
    CallSite(tr::T, fn, ret::K) where {T <: Trace, K} = new{T, Tuple{}, K}(tr, fn, (), ret)
    CallSite(tr::T, fn, args::J, ret::K) where {T <: Trace, J, K} = new{T, J, K}(tr, fn, args, ret)
end

# Hierarchical - standard interpreter style.
mutable struct HierarchicalTrace <: Trace
    chm::Dict{Address, RecordSite}
    score::Float64
    HierarchicalTrace() = new(Dict{Address, ChoiceSite}(), 0.0)
end


# Graph - for when analysis is available.
mutable struct GraphTrace <: Trace
    sub::Dict{Address, RecordSite}
    dependencies::Dict{Address, Vector{Address}}
    nocacheable::Vector{DataType}
    cache::IdDict
    score::Float64
    GraphTrace() = new(Dict{Address, RecordSite}(),
                       Dict{Address, Vector{Address}}(),
                       [typeof(rand)],
                       IdDict(),
                       score::Float64)
end

Trace() = HierarchicalTrace()
