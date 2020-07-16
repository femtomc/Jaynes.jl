import Base.getindex
import Base: rand

# ------------ Core ------------ #

# Special calls recognized by tracer.
rand(addr::Address, d::Distribution{T}) where T = rand(d)
rand(addr::Address, fn::Function, args...) = fn(args...)
learnable(addr::Address, p::T) where T = p
plate(addr::Address, args...) = error("(plate) call with address $addr evaluated outside of the tracer.")
markov(addr::Address, args...) = error("(markov) call with address $addr evaluated outside of the tracer.")

# Record sites for choices, calls, and parameters.
abstract type RecordSite end
abstract type CallSite <: RecordSite end
abstract type LearnableSite <: RecordSite end

struct ChoiceSite{T} <: RecordSite
    score::Float64
    val::T
end
get_score(cs::ChoiceSite) = cs.score

struct ParameterSite{T} <: LearnableSite
    val::T
end

# ------------ Hierarchical trace ------------ #

abstract type Trace end

@dynamo function (tr::Trace)(a...)
    ir = IR(a...)
    ir == nothing && return
    recur!(ir)
    return ir
end

struct HierarchicalTrace <: Trace
    calls::Dict{Address, CallSite}
    choices::Dict{Address, ChoiceSite}
    params::Dict{Address, LearnableSite}
    function HierarchicalTrace()
        new(Dict{Address, CallSite}(), 
            Dict{Address, ChoiceSite}(),
            Dict{Address, LearnableSite}())
    end
end
Trace() = HierarchicalTrace()
has_choice(tr::HierarchicalTrace, addr) = haskey(tr.choices, addr)
has_call(tr::HierarchicalTrace, addr::Address) = haskey(tr.calls, addr)
get_call(tr::HierarchicalTrace, addr::Address) = tr.calls[addr]
get_choice(tr::HierarchicalTrace, addr) = tr.choices[addr]
get_param(tr::HierarchicalTrace, addr) = tr.params[addr].val
function get_call(tr::HierarchicalTrace, addr::Pair)
    get_call(tr.calls[addr[1]], addr[2])
end
add_call!(tr::HierarchicalTrace, addr, cs::T) where T <: CallSite = tr.calls[addr] = cs
add_choice!(tr::HierarchicalTrace, addr, cs::ChoiceSite) = tr.choices[addr] = cs

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

# ------------ Branch trace ------------ #

mutable struct BranchTrace{T <: RecordSite, B <: RecordSite} <: Trace
    condtrace::T
    branchtrace::B
    params::Dict{Address, LearnableSite}
end

# ------------ Call sites ------------ #

# Black-box
mutable struct GenericCallSite{J, K} <: CallSite
    trace::HierarchicalTrace
    score::Float64
    fn::Function
    args::J
    ret::K
end
has_choice(bbcs::GenericCallSite, addr) = haskey(bbcs.tr.choices, addr)
has_call(bbcs::GenericCallSite, addr) = haskey(bbcs.tr.calls, addr)
get_call(bbcs::GenericCallSite, addr) = bbcs.tr.calls[addr]
get_score(bbcs::GenericCallSite) = bbcs.score

# Vectorized
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

# If-else branch site
mutable struct ConditionalBranchSite{C, A, B, T <: RecordSite, K <: RecordSite, J, L, R}
    trace::BranchTrace
    score::Float64
    cond_kernel::C
    cond_args::J
    a::A
    b::B
    branch_args::L
    ret::R
end

# ------------ getindex ------------ #

getindex(cs::ChoiceSite, addr::Address) = nothing
getindex(cs::GenericCallSite, addr) = getindex(cs.trace, addr)
getindex(vcs::VectorizedSite, addr::Int) = getindex(cs.trace, addr)
function getindex(vcs::VectorizedSite, addr::Pair)
    getindex(vcs.trace[addr[1]], addr[2])
end
unwrap(cs::ChoiceSite) = cs.val
unwrap(cs::GenericCallSite) = cs.ret
unwrap(cs::VectorizedSite) = cs.ret
function getindex(tr::HierarchicalTrace, addr::Address)
    if has_choice(tr, addr)
        return unwrap(get_choice(tr, addr))
    else
        return nothing
    end
end
function getindex(tr::HierarchicalTrace, addr::Pair)
    if has_call(tr, addr[1])
        return getindex(get_call(tr, addr[1]), addr[2])
    else
        return nothing
    end
end

# ------------ haskey ------------ #

import Base.haskey
haskey(cs::ChoiceSite, addr::Address) = false
haskey(cs::GenericCallSite, addr) = haskey(cs.trace, addr)
function haskey(vcs::VectorizedSite, addr::Pair)
    addr[1] <= length(vcs.trace.subrecords) && haskey(vcs.trace[addr[1]], addr[2])
end
function Base.haskey(tr::HierarchicalTrace, addr::Address)
    has_choice(tr, addr)
end
function Base.haskey(tr::HierarchicalTrace, addr::Pair)
    if has_call(tr, addr[1])
        return Base.haskey(get_call(tr, addr[1]), addr[2])
    else
        return false
    end
end

# ------------ Documentation ------------ #

@doc(
"""
```julia
abstract type Trace end
```
Abstract base type of all traces.
""", Trace)

@doc(
"""
```julia
mutable struct HierarchicalTrace <: Trace
    calls::Dict{Address, CallSite}
    choices::Dict{Address, ChoiceSite}
    params::Dict{Address, LearnableSite}
    end
end
```
Structured execution trace with tracked randomness in a function call.
""", HierarchicalTrace)

@doc(
"""
```julia
struct ChoiceSite{T} <: RecordSite
    score::Float64
    val::T
end
```
A record of a random sample at an addressed `rand` call with a `Distribution` instance. Keeps the value of the random sample and the `logpdf` score.
""", ChoiceSite)

@doc(
"""
```julia
abstract type CallSite <: RecordSite end
```
Abstract base type of all call sites.
""", CallSite)

@doc(
"""
```julia
mutable struct GenericCallSite{J, K} <: CallSite
    trace::HierarchicalTrace
    score::Float64
    fn::Function
    args::J
    ret::K
end
```
A record of a black-box call (e.g. no special tracer language features). Records the `fn` and `args` for the call, as well as the `ret` return value.
""", GenericCallSite)

@doc(
"""
```julia
mutable struct VectorizedSite{F <: Function, C <: RecordSite, J, K} <: CallSite
    trace::VectorizedTrace{C}
    score::Float64
    fn::Function
    args::J
    ret::Vector{K}
end
```
A record of a call site using the special `plate` and `markov` tracer language features. Informs the tracer that the call conforms to a special pattern of randomness dependency, which allows the storing of `Trace` instances sequentially in a vector.
""", VectorizedSite)
