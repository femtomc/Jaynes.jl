# ------------ Vectorized trace ------------ #

struct VectorizedTrace{C <: RecordSite} <: Trace
    subrecords::Vector{C}
    VectorizedTrace{C}() where C = new{C}(Vector{C}())
    VectorizedTrace(arr::Vector{C}) where C <: RecordSite = new{C}(arr)
end

has_choice(tr::VectorizedTrace{<: CallSite}, addr::Int) = false
function has_choice(tr::VectorizedTrace{<: CallSite}, addr::T) where T <: Tuple
    isempty(addr) && return false
    length(addr) == 1 && return has_choice(tr, addr[1])
    has_call(tr, addr[1]) && has_choice(get_call(tr, addr[1]), addr[2 : end])
end
has_choice(tr::VectorizedTrace{ChoiceSite}, addr::Int) = addr < length(tr.subrecords)

get_choice(tr::VectorizedTrace{<: ChoiceSite}, addr::Int) = return tr.subrecords[addr]
function get_choice(tr::VectorizedTrace{<: CallSite}, addr::T) where T <: Tuple
    isempty(addr) && error("VectorizedTrace (get_choice): vectorized trace contains call sites, no choice sites at $addr.")
    length(addr) == 1 && return get_choice(tr, addr[1])
    return get_choice(get_call(tr, addr[1]), addr[2 : end])
end

has_call(tr::VectorizedTrace{<: ChoiceSite}, addr) = false
has_call(tr::VectorizedTrace{<: CallSite}, addr::Int) = addr <= length(tr.subrecords)
function has_call(tr::VectorizedTrace{<: CallSite}, addr::T) where T <: Tuple
    isempty(addr) && return false
    length(addr) == 1 && return has_call(tr, addr[1])
    has_call(tr, addr[1]) && has_call(get_call(tr, addr[1]), addr[2 : end])
end

Base.haskey(tr::VectorizedTrace{<: ChoiceSite}, addr::T) where T <: Address = has_choice(tr, addr)
Base.haskey(tr::VectorizedTrace{<: CallSite}, addr::T) where T <: Address = has_call(tr, addr)
function Base.haskey(tr::VectorizedTrace{<: CallSite}, addr::Tuple) where T <: Address
    isempty(addr) && return false
    length(addr) == 1 && return haskey(tr, addr[1])
    has_call(tr, addr[1]) && haskey(get_call(tr, addr[1]), addr[2 : end])
end

get_call(tr::VectorizedTrace{<: CallSite}, addr::Int) = return tr.subrecords[addr]
function get_call(tr::VectorizedTrace{<: CallSite}, addr::T) where T <: Tuple
    return tr.subrecords[addr]
end

add_call!(tr::VectorizedTrace{<: CallSite}, cs::CallSite) = push!(tr.subrecords, cs)

add_choice!(tr::VectorizedTrace{<: ChoiceSite}, cs::ChoiceSite) = push!(tr.subrecords, cs)

Base.getindex(vt::VectorizedTrace, addr::Int) = vt.subrecords[addr]
function getindex(vt::VectorizedTrace, addrs...)
    has_call(vt, addrs) && return get_call(vt, addrs)
    has_choice(vt, addrs) && return get_choice(vt, addrs)
    error("VectorizedTrace (getindex): no choice or call at $addrs")
end

# ------------ Utility used for pretty printing ------------ #

function collect!(par::T, addrs::Vector{Any}, chd::Dict{Any, Any}, tr::VectorizedTrace, meta) where T <: Tuple
    for (k, v) in enumerate(tr.subrecords)
        if v isa ChoiceSite
            push!(addrs, (par..., k))
            chd[(par..., k)] = v.val
        elseif v isa HierarchicalCallSite
            collect!((par..., k), addrs, chd, v.trace, meta)
        elseif v isa VectorizedCallSite
            for i in 1:length(v.trace.subrecords)
                collect!((par..., k, i), addrs, chd, v.trace.subrecords[i].trace, meta)
            end
        end
    end
end
function collect!(addrs::Vector{Any}, chd::Dict{Any, Any}, tr::VectorizedTrace, meta)
    for (k, v) in enumerate(tr.subrecords)
        if v isa ChoiceSite
            push!(addrs, (k, ))
            chd[k] = v.val
        elseif v isa HierarchicalCallSite
            collect!((k, ), addrs, chd, v.trace, meta)
        elseif v isa VectorizedCallSite
            collect!((k, i), addrs, chd, v.trace, meta)
        end
    end
end

# ------------ Vectorized site ------------ #

struct VectorizedCallSite{F, D, C <: RecordSite, J, K} <: CallSite
    trace::VectorizedTrace{C}
    score::Float64
    kernel::D
    args::J
    ret::Vector{K}
    function VectorizedCallSite{F}(sub::VectorizedTrace{C}, sc::Float64, kernel::D, args::J, ret::Vector{K}) where {F, D, C <: RecordSite, J, K}
        new{F, D, C, J, K}(sub, sc, kernel, args, ret)
    end
end

has_choice(vcs::VectorizedCallSite, addr) = has_choice(vcs.trace, addr)

get_choice(vcs::VectorizedCallSite, addr) = get_choice(vcs.trace, addr)

has_call(vcs::VectorizedCallSite, addr) = has_call(vcs.trace, addr)

get_call(vcs::VectorizedCallSite, addr) = get_call(vcs.trace, addr)

get_score(vcs::VectorizedCallSite) = vcs.score

getindex(vcs::VectorizedCallSite, addrs...) = getindex(vcs.trace, addrs...)

haskey(vcs::VectorizedCallSite, addrs...) = haskey(vcs.trace, addrs...)

get_ret(cs::VectorizedCallSite) = cs.ret

# ------------ Vectorized discard trace ------------ #

struct VectorizedDiscard <: Trace
    subrecords::Dict{Int, RecordSite}
    VectorizedDiscard() = new(Dict{Int, RecordSite}())
end

get_call(tr::VectorizedDiscard, addr) = return tr.subrecords[addr]

add_call!(tr::VectorizedDiscard, addr::Int, cs::CallSite) = tr.subrecords[addr] = cs

Base.getindex(vt::VectorizedDiscard, addr::Int) = vt.subrecords[addr]
