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
has_choice(tr::HierarchicalTrace, addr) = haskey(tr.choices, addr)
has_call(tr::HierarchicalTrace, addr::Address) = haskey(tr.calls, addr)
get_call(tr::HierarchicalTrace, addr::Address) = tr.calls[addr]
get_choice(tr::HierarchicalTrace, addr) = tr.choices[addr]
function get_call(tr::HierarchicalTrace, addr::Pair)
    get_call(tr.calls[addr[1]], addr[2])
end
add_call!(tr::HierarchicalTrace, addr, cs::T) where T <: CallSite = tr.calls[addr] = cs
add_choice!(tr::HierarchicalTrace, addr, cs::ChoiceSite) = tr.choices[addr] = cs
function getindex(tr::HierarchicalTrace, addr::Address)
    has_choice(tr, addr) && return unwrap(get_choice(tr, addr))
    return nothing
end
function getindex(tr::HierarchicalTrace, hd::T, next::K, addrs...) where {K, T <: Address}
    has_call(tr, hd) && return getindex(get_call(tr, hd), next, addrs...)
    return nothing
end
function Base.haskey(tr::HierarchicalTrace, addr::Address)
    has_choice(tr, addr)
end
function Base.haskey(tr::HierarchicalTrace, addr::Tuple)
    fst = addr[1]
    tl = addr[2:end]
    isempty(tl) && begin
        return Base.haskey(tr, fst)
    end
    if has_call(tr, fst)
        return Base.haskey(get_call(tr, fst), tl)
    else
        return false
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
unwrap(cs::HierarchicalCallSite) = cs.ret
