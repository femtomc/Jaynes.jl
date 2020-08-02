# ------------ Hierarchical trace ------------ #

struct HierarchicalTrace <: Trace
    calls::Dict{Address, CallSite}
    choices::Dict{Address, ChoiceSite}
    function HierarchicalTrace()
        new(Dict{Address, CallSite}(), 
            Dict{Address, ChoiceSite}())
    end
end

Trace() = HierarchicalTrace()

has_top(tr::HierarchicalTrace, addr::T) where T <: Address = haskey(tr.choices, addr)
function has_top(tr::HierarchicalTrace, addr::T) where T <: Tuple
    isempty(addr) && return false
    length(addr) == 1 && return has_top(tr, addr[1])
    has_sub(tr, addr[1]) && has_top(get_sub(tr, addr[1]), addr[2 : end])
end

has_sub(tr::HierarchicalTrace, addr::T) where T <: Address = haskey(tr.calls, addr)
function has_sub(tr::HierarchicalTrace, addr::T) where T <: Tuple
    isempty(addr) && return false
    length(addr) == 1 && return has_sub(tr, addr[1])
    has_sub(tr, addr[1]) && has_sub(get_sub(tr, addr[1]), addr[2 : end])
end

get_sub(tr::HierarchicalTrace, addr::T) where T <: Address = tr.calls[addr]
function get_sub(tr::HierarchicalTrace, addr::T) where T <: Tuple
    isempty(addr) && error("HierarchicalTrace (get_sub): no call in trace at $addr.")
    length(addr) == 1 && return get_sub(tr, addr[1])
    get_sub(get_sub(tr, addr[1]), addr[2 : end])
end

get_top(tr::HierarchicalTrace, addr::T) where T <: Address = tr.choices[addr]
function get_top(tr::HierarchicalTrace, addr::T) where T <: Tuple
    isempty(addr) && error("HierarchicalTrace (get_top): no choice in trace at $addr.")
    length(addr) == 1 && return get_top(tr, addr[1])
    get_top(get_sub(tr, addr[1]), addr[2 : end])
end

function getindex(tr::HierarchicalTrace, addrs...)
    has_sub(tr, addrs) && return get_sub(tr, addrs)
    has_top(tr, addrs) && return get_top(tr, addrs)
    error("HierarchicalTrace (getindex): no choice or call at $addrs.")
end

function Base.haskey(tr::HierarchicalTrace, addr::T) where T <: Address
    has_top(tr, addr) && return true
    has_sub(tr, addr) && return true
    return false
end
function Base.haskey(tr::HierarchicalTrace, addr::T) where T <: Tuple
    isempty(addr) && return false
    length(addr) == 1 && return haskey(tr, addr[1])
    has_sub(tr, addr[1]) && haskey(get_sub(tr, addr[1]), addr[2 : end])
end

# These methods only work for addresses. You should never be adding a call or choice site, except at the current stack level.
add_call!(tr::HierarchicalTrace, addr::T, cs::K) where {T <: Address, K <: CallSite} = tr.calls[addr] = cs
add_choice!(tr::HierarchicalTrace, addr::T, cs::ChoiceSite) where T <: Address = tr.choices[addr] = cs

# ------------- Utility for pretty printing ------------ $

function collect!(par::T, addrs::Vector{Any}, chd::Dict{Any, Any}, tr::HierarchicalTrace, meta) where T <: Tuple
    for (k, v) in tr.choices
        push!(addrs, (par..., k))
        chd[(par..., k)] = v.val
    end
    for (k, v) in tr.calls
        collect!((par..., k), addrs, chd, v.trace, meta)
    end
end

function collect!(addrs::Vector{Any}, chd::Dict{Any, Any}, tr::HierarchicalTrace, meta)
    for (k, v) in tr.choices
        push!(addrs, (k, ))
        chd[(k, )] = v.val
    end
    for (k, v) in tr.calls
        collect!((k, ), addrs, chd, v.trace, meta)
    end
end

# ------------ Hierarchical call site ------------ #

struct HierarchicalCallSite{J, K} <: CallSite
    trace::HierarchicalTrace
    score::Float64
    fn::Function
    args::J
    ret::K
end

has_top(bbcs::HierarchicalCallSite, addr) = has_top(bbcs.trace, addr)

get_top(bbcs::HierarchicalCallSite, addr) = get_top(bbcs.trace, addr)

has_sub(bbcs::HierarchicalCallSite, addr) = has_sub(bbcs.trace, addr)

get_sub(bbcs::HierarchicalCallSite, addr) = get_sub(bbcs.trace, addr)

get_score(bbcs::HierarchicalCallSite) = bbcs.score

getindex(cs::HierarchicalCallSite, addrs...) = getindex(cs.trace, addrs...)

haskey(cs::HierarchicalCallSite, addr) = haskey(cs.trace, addr)

get_ret(cs::HierarchicalCallSite) = cs.ret
