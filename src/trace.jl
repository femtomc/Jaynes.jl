import Base: rand
rand(addr::Address, d::Distribution{T}) where T = rand(d)
rand(addr::Address, fn::Function, args...) = fn(args...)

abstract type RecordSite end
abstract type Trace <: ExecutionContext end

# These are the core tracing data structures which represent random choices, function call sites, and the trace of execution in any particular context.
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

abstract type CallSite <: RecordSite end
mutable struct BlackBoxCallSite{T <: Trace, J, K} <: CallSite
    trace::T
    fn::Function
    args::J
    ret::K
end

mutable struct VectorizedCallSite{T <: Trace, J, K} <: CallSite
    subtraces::Vector{T}
    score::Float64
    fn::Function
    args::J
    ret::Vector{K}
end

score(tr::T) where T <: Trace = tr.score
score(cs::BlackBoxCallSite) = cs.trace.score
score(vcs::VectorizedCallSite) = vcs.score

# ------------ Direct execution with trace ------------ #

@inline function (tr::HierarchicalTrace)(fn::typeof(rand), addr::Address, d::Distribution{T}) where T
    s = rand(d)
    tr.chm[addr] = ChoiceSite(logpdf(d, s), s)
    return s
end

@inline function (tr::HierarchicalTrace)(fn::typeof(rand), addr::Address, call::Function, args...)
    n_tr = Trace()
    ret = n_tr(call, args...)
    tr.chm[addr] = BlackBoxCallSite(n_tr, call, args, ret)
    return ret
end

# Vectorized foldr call.
@inline function (tr::HierarchicalTrace)(c::typeof(foldr), fn::typeof(rand), addr::Address, call::Function, len::Int, args...)
    n_tr = Trace()
    ret = n_tr(call, args...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_tr = Vector{HierarchicalTrace}(undef, len)
    v_ret[1] = ret
    v_tr[1] = n_tr
    for i in 2:len
        n_tr = Trace()
        ret = n_tr(call, v_ret[i-1]...)
        v_ret[i] = ret
        v_tr[i] = n_tr
    end
    score = sum(map(v_tr) do tr
                    score(tr)
                end)
    tr.chm[addr] = VectorizedCallSite(v_tr, score, fn, args, v_ret)
    return v_ret
end

# Vectorized map call.
@inline function (tr::HierarchicalTrace)(c::typeof(map), fn::typeof(rand), addr::Address, call::Function, args::Vector)
    n_tr = Trace()
    ret = n_tr(call, args[1]...)
    len = length(args)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_tr = Vector{HierarchicalTrace}(undef, len)
    v_ret[1] = ret
    v_tr[1] = n_tr
    for i in 2:len
        n_tr = Trace()
        ret = n_tr(call, args[i]...)
        v_ret[i] = ret
        v_tr[i] = n_tr
    end
    score = sum(map(v_tr) do tr
                    score(tr)
                end)
    tr.chm[addr] = VectorizedCallSite(v_tr, score, fn, args, v_ret)
    return v_ret
end

# Allows instances of Trace and CallSite to be accessed through indexing like a dictionary. 
# setindex! is not defined - we treat the trace as immutable, unless you interact with it through contexts for inference.
import Base.getindex
getindex(cs::ChoiceSite, addr::Address) = nothing
getindex(cs::BlackBoxCallSite, addr) = getindex(cs.trace, addr)
getindex(vcs::VectorizedCallSite, addr::Int) = cs.subtraces[addr]
function getindex(vcs::VectorizedCallSite, addr::Pair)
    getindex(vcs.subtraces[addr[1]], addr[2])
end
unwrap(cs::ChoiceSite) = cs.val
unwrap(cs::BlackBoxCallSite) = cs.ret
unwrap(cs::VectorizedCallSite) = cs.ret
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
haskey(cs::BlackBoxCallSite, addr) = haskey(cs.trace, addr)
function haskey(vcs::VectorizedCallSite, addr::Pair)
    addr[1] < length(vcs.subtraces) && haskey(vcs.subtraces, addr[2])
end
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

