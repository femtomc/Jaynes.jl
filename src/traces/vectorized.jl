# ------------ Vectorized trace ------------ #

mutable struct VectorizedTrace{C <: RecordSite} <: Trace
    subrecords::Vector{C}
    params::Dict{Address, LearnableSite}
    VectorizedTrace{C}() where C = new{C}(Vector{C}(), Dict{Address, LearnableSite}())
    VectorizedTrace(arr::Vector{C}) where C <: RecordSite = new{C}(arr, Dict{Address, LearnableSite}()) 
end
has_choice(tr::VectorizedTrace{<: CallSite}, addr) = false
function has_choice(tr::VectorizedTrace{ChoiceSite}, addr)
    return addr < length(tr.subrecords)
end
has_call(tr::VectorizedTrace{<: ChoiceSite}, addr) = false
has_call(tr::VectorizedTrace{<: CallSite}, addr) = false
function has_call(tr::VectorizedTrace{<: CallSite}, addr::Int)
    return addr <= length(tr.subrecords)
end
get_call(tr::VectorizedTrace{<: CallSite}, addr) = return tr.subrecords[addr]
add_call!(tr::VectorizedTrace{<: CallSite}, cs::CallSite) = push!(tr.subrecords, cs)
Base.getindex(vt::VectorizedTrace, addr::Int) = vt.subrecords[addr]

# ------------ Vectorized site ------------ #

mutable struct VectorizedSite{F, D, C <: RecordSite, J, K} <: CallSite
    trace::VectorizedTrace{C}
    score::Float64
    kernel::D
    args::J
    ret::Vector{K}
    function VectorizedSite{F}(sub::VectorizedTrace{C}, sc::Float64, kernel::D, args::J, ret::Vector{K}) where {F, D, C <: RecordSite, J, K}
        new{F, D, C, J, K}(sub, sc, kernel, args, ret)
    end
end
function has_choice(vcs::VectorizedSite, addr)
    has_choice(vcs.trace, addr) && return true
    return false
end
function has_call(vcs::VectorizedSite, addr)
    has_call(vcs.trace, addr) && return true
    return false
end
function get_call(vcs::VectorizedSite, addr)
    has_call(vcs.trace, addr) && return get_call(vcs.trace, addr)
    error("VectorizedSite (get_call): no call at $addr.")
end
get_score(vcs::VectorizedSite) = vcs.score
getindex(vcs::VectorizedSite, addr::Int) = getindex(cs.trace, addr)
function getindex(vcs::VectorizedSite, addr::Pair)
    getindex(vcs.trace[addr[1]], addr[2])
end
function haskey(vcs::VectorizedSite, addr::Pair)
    addr[1] <= length(vcs.trace.subrecords) && haskey(vcs.trace[addr[1]], addr[2])
end
unwrap(cs::VectorizedSite) = cs.ret

# ------------ Vectorized discard trace ------------ #

mutable struct VectorizedDiscard <: Trace
    subrecords::Dict{Int, RecordSite}
    VectorizedDiscard() = new(Dict{Int, RecordSite}())
end
get_call(tr::VectorizedDiscard, addr) = return tr.subrecords[addr]
add_call!(tr::VectorizedDiscard, addr::Int, cs::CallSite) = tr.subrecords[addr] = cs
Base.getindex(vt::VectorizedDiscard, addr::Int) = vt.subrecords[addr]
