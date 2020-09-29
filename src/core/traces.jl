# ------------ Core ------------ #

Base.iterate(s::Symbol) = s

# Special calls with fallbacks.
@inline rand(addr::A, d::Distribution{T}) where {A <: Address, T} = rand(d)
@inline rand(addr::A, fn::Function, args...) where A <: Address = fn(args...)
@inline rand(addr::A, fn, args...) where A <: Address = fn(args...)

# Special features - must be evaluated by context tracer.
learnable(addr::A) where {A <: Address} = error("(learnable) call with address $addr evaluated outside of a context tracer.\nThis normally occurs because you're not matching dispatch correctly.")
fillable(addr::A) where {A <: Address} = error("(fillable) call with address $addr evaluated outside of a context tracer.\nThis normally occurs because you're not matching dispatch correctly.")
factor(args...) = args

# ------------ Call sites ------------ #

abstract type CallSite <: AddressMap{Choice} end

@inline get_trace(cs::C) where C <: CallSite = cs.trace
@inline has_value(cs::C, addr) where C <: CallSite = has_value(get_trace(cs), addr)
@inline get_value(cs::C, addr) where C <: CallSite = get_value(get_sub(get_trace(cs), addr))
@inline get_sub(cs::C, addr::A) where {C <: CallSite, A <: Address} = get_sub(get_trace(cs), addr)
@inline get_sub(cs::C, addr::Tuple{}) where {C <: CallSite, A <: Address} = get_sub(get_trace(cs), addr)
@inline get_sub(cs::C, addr::Tuple{A}) where {C <: CallSite, A <: Address} = get_sub(get_trace(cs), addr)
@inline get_sub(cs::C, addr::Tuple) where C <: CallSite = get_sub(get_trace(cs), addr::Tuple)
@inline get_score(cs::C) where C <: CallSite = cs.score
@inline get_ret(cs::C) where C <: CallSite = cs.ret
@inline get_value(cs::C) where C <: CallSite = get_ret(cs)
@inline get_args(cs::C) where C <: CallSite = cs.args
@inline get_choices(cs::C) where C <: CallSite = get_choices(get_trace(cs))
@inline has_sub(cs::C, addr::A) where {C <: CallSite, A <: Address} = has_sub(get_trace(cs), addr)
@inline haskey(cs::C, addr::A) where {C <: CallSite, A <: Address} = haskey(get_trace(cs), addr)
@inline getindex(cs::C, addrs...) where C <: CallSite = getindex(get_trace(cs), addrs...)
@inline shallow_iterator(cs::C) where C <: CallSite = shallow_iterator(get_trace(cs))
@inline collect!(par::T, addrs::Vector, chd::Dict, cs::C, meta) where {T <: Tuple, C <: CallSite} = collect!(par, addrs, chd, get_trace(cs), meta)
@inline flatten(cs::C) where C <: CallSite = flatten(get_trace(cs))
@inline merge(cs::C, am::A) where {C <: CallSite, A <: AddressMap} = merge(get_trace(cs), am)
@inline merge!(cs::C, am::A) where {C <: CallSite, A <: AddressMap} = merge!(get_trace(cs), am)
@inline target(cs::C, arr::Vector) where C <: CallSite = target(get_trace(cs), arr)

function Base.display(call::C; 
                      fields::Array{Symbol, 1} = [:val],
                      show_full = false) where C <: CallSite
    println("  __________________________________\n")
    println("               Playback\n")
    println(" type : $C\n")
    map(fieldnames(C)) do f
        val = getfield(call, f)
        typeof(val) <: Real && begin
            println(" $(f) : $(val)\n")
            return
        end
        println(" $(f) : $(typeof(val))\n")
    end
    println("  __________________________________\n")
end

# ------------ includes ------------ #

include("traces/dynamic.jl")

# ------------ Documentation ------------ #

@doc(
"""
```julia
rand(addr::A, d::Distribution{T}) where {A <: Address, T}
rand(addr::A, fn::Function, args...) where A <: Address
```

A `rand` call with a first argument which is a subtype of `Address` informs the context context tracers that this call should be intercepted and recorded. If the call includes a distribution as final argument, a context tracer will create a `ChoiceSite` representation and reason about the call accordingly. If the call includes a function and a set of arguments, a context tracer will create a `HierarchicalCallSite` representation and recurse into the call.
""", rand)

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
abstract type Trace end
```
Abstract base type of all traces.
""", Trace)

@doc(
"""
```julia
struct HierarchicalTrace <: Trace
    calls::Dict{Address, CallSite}
    choices::Dict{Address, ChoiceSite}
    params::Dict{Address, LearnableSite}
    end
end
```
Execution trace which allows the tracking of randomness metadata in a function call.
""", HierarchicalTrace)

@doc(
"""
```julia
struct VectorizedTrace{C <: RecordSite} <: Trace
    subrecords::Vector{C}
    params::Dict{Address, LearnableSite}
end

```
Structured execution trace for `markov` and `plate` calls. The dependency structure interpretation of the `subrecords` vector depends on the call. For `markov`, the structure is Markovian. For `plate`, each element is drawn IID from the program or distribution provided to `plate`.
""", VectorizedTrace)

@doc(
"""
```julia
struct HierarchicalCallSite{J, K} <: CallSite
    trace::HierarchicalTrace
    score::Float64
    fn::Function
    args::J
    ret::K
end
```
A record of a black-box call (e.g. no special context tracer language features). Records the `fn` and `args` for the call, as well as the `ret` return value.
""", HierarchicalCallSite)
