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

# Allows instances of Trace and CallSite to be accessed through indexing like a dictionary. 
# setindex! is not defined - we treat the trace as immutable, unless you interact with it through contexts for inference.
import Base.getindex
getindex(cs::ChoiceSite, addr::Address) = nothing
getindex(cs::CallSite, addr) = getindex(cs.trace, addr)
unwrap(cs::ChoiceSite) = cs.val
unwrap(cs::CallSite) = cs.ret
function getindex(tr::HierarchicalTrace, addr::Address)
    if haskey(tr.chm, addr)
        return unwrap(tr.chm[addr])
    else
        return nothing
    end
end
function getindex(tr::HierarchicalTrace, addr::Pair)
    if haskey(tr.chm, addr[1])
        return getindex(tr.chm[addr[1]], addr[2])
    else
        return nothing
    end
end

import Base.haskey
haskey(cs::ChoiceSite, addr::Address) = false
haskey(cs::CallSite, addr) = haskey(cs.trace, addr)
function Base.haskey(tr::HierarchicalTrace, addr::Address)
    haskey(tr.chm, addr)
end
function Base.haskey(tr::HierarchicalTrace, addr::Pair)
    if haskey(tr.chm, addr[1])
        return haskey(tr.chm[1], addr[2])
    else
        return false
    end
end
