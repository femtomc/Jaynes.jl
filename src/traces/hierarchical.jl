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

has_choice(tr::HierarchicalTrace, addr::T) where T <: Address = haskey(tr.choices, addr)
function has_choice(tr::HierarchicalTrace, addr::T) where T <: Tuple
    isempty(addr) && return false
    length(addr) == 1 && return has_choice(tr, addr[1])
    has_call(tr, addr[1]) && return has_choice(get_call(tr, addr[1]), addr[2 : end])
end

has_call(tr::HierarchicalTrace, addr::T) where T <: Address = haskey(tr.calls, addr)
function has_call(tr::HierarchicalTrace, addr::T) where T <: Tuple
    isempty(addr) && return false
    length(addr) == 1 && return has_call(tr, addr[1])
    has_call(tr, addr[1]) && return has_call(get_call(tr, addr[1]), addr[2 : end])
end

get_call(tr::HierarchicalTrace, addr::T) where T <: Address = tr.calls[addr]
function get_call(tr::HierarchicalTrace, addr::T) where T <: Tuple
    isempty(addr) && error("HierarchicalTrace (get_call): no call in trace at $addr.")
    length(addr) == 1 && return get_call(tr, addr[1])
    get_call(get_call(tr, addr[1]), addr[2 : end])
end

get_choice(tr::HierarchicalTrace, addr::T) where T <: Address = tr.choices[addr]
function get_choice(tr::HierarchicalTrace, addr::T) where T <: Tuple
    isempty(addr) && error("HierarchicalTrace (get_choice): no choice in trace at $addr.")
    length(addr) == 1 && return get_choice(tr, addr[1])
    get_choice(get_call(tr, addr[1]), addr[2 : end])
end

function getindex(tr::HierarchicalTrace, addr...)
    has_call(tr, addr) && return get_call(tr, addr)
    has_choice(tr, addr) && return get_choice(tr, addr)
    error("HierarchicalTrace (getindex): no choice or call at $addr.")
end

function Base.haskey(tr::HierarchicalTrace, addr::T) where T <: Address
    has_choice(tr, addr) && return true
    has_call(tr, addr) && return true
    return false
end
function Base.haskey(tr::HierarchicalTrace, addr::T) where T <: Tuple
    isempty(addr) && return false
    length(addr) == 1 && return haskey(tr, addr[1])
    has_call(tr, addr[1]) && return haskey(get_call(tr, addr[1]), addr[2 : end])
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
        if v isa HierarchicalCallSite
            collect!((par..., k), addrs, chd, v.trace, meta)
        elseif v isa VectorizedCallSite
            for i in 1:length(v.trace.subrecords)
                collect!((par..., k, i), addrs, chd, v.trace.subrecords[i].trace, meta)
            end
        end
    end
end

function collect!(addrs::Vector{Any}, chd::Dict{Any, Any}, tr::HierarchicalTrace, meta)
    for (k, v) in tr.choices
        push!(addrs, (k, ))
        chd[(k, )] = v.val
    end
    for (k, v) in tr.calls
        if v isa HierarchicalCallSite
            collect!((k, ), addrs, chd, v.trace, meta)
        elseif v isa VectorizedCallSite
            collect!((k, ), addrs, chd, v.trace, meta)
        end
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

has_choice(bbcs::HierarchicalCallSite, addr) = haskey(bbcs.trace.choices, addr)

has_call(bbcs::HierarchicalCallSite, addr) = haskey(bbcs.trace.calls, addr)

get_call(bbcs::HierarchicalCallSite, addr) = get_call(bbcs.trace, addr)

get_score(bbcs::HierarchicalCallSite) = bbcs.score

getindex(cs::HierarchicalCallSite, addrs...) = getindex(cs.trace, addrs...)

haskey(cs::HierarchicalCallSite, addr) = haskey(cs.trace, addr)

get_ret(cs::HierarchicalCallSite) = cs.ret
