import Base: rand
rand(addr::Address, d::Distribution{T}) where T = rand(d)
rand(addr::Address, fn::Function, args...) = fn(args...)

abstract type RecordSite end
abstract type Trace <: ExecutionContext end

mutable struct HierarchicalTrace <: Trace
    chm::Dict{Address, RecordSite}
    score::Float64
    function HierarchicalTrace()
        new(Dict{Address, RecordSite}(), 0.0)
    end
end
Trace() = HierarchicalTrace()

struct ChoiceSite{T} <: RecordSite
    score::Float64
    val::T
end

mutable struct CallSite{T <: Trace, J, K} <: RecordSite
    trace::T
    fn::Function
    args::J
    ret::K
end

mutable struct VectorizedCallSite{T <: Trace, J, K} <: RecordSite
    subtraces::Vector{T}
    fn::Function
    args::J
    ret::Vector{K}
end

@inline function (tr::HierarchicalTrace)(fn::typeof(rand), addr::Address, d::Distribution{T}) where T
    s = rand(d)
    tr.chm[addr] = ChoiceSite(logpdf(d, s), s)
    return s
end

@inline function (tr::HierarchicalTrace)(fn::typeof(rand), addr::Address, call::Function, args...)
    n_tr = Trace()
    ret = n_tr(call, args...)
    tr.chm[addr] = CallSite(n_tr, call, args, ret)
    return ret
end

# Vectorized call.
@inline function (tr::HierarchicalTrace)(c::typeof(foldr), fn::typeof(rand), addr::Address, call::Function, args...)
    n_tr = Trace()
    ret = n_tr(call, args...)
    tr.chm[addr] = CallSite(n_tr, call, args, ret)
    return ret
end

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

