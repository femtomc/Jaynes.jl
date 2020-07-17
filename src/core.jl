import Base: getindex, haskey, rand

# ------------ Core ------------ #

# Special calls recognized by tracer.
rand(addr::Address, d::Distribution{T}) where T = rand(d)
rand(addr::Address, fn::Function, args...) = fn(args...)
learnable(addr::Address, p::T) where T = p
plate(addr::Address, args...) = error("(plate) call with address $addr evaluated outside of the tracer.")
markov(addr::Address, args...) = error("(markov) call with address $addr evaluated outside of the tracer.")
cond(addr::Address, args...) = error("(cond) call with address $addr evaluated outside of the tracer.")

# Generic abstract types..
abstract type RecordSite end
abstract type CallSite <: RecordSite end
abstract type LearnableSite <: RecordSite end

# ------------ Choice sites - rand calls with distributions. ------------ #

struct ChoiceSite{T} <: RecordSite
    score::Float64
    val::T
end
get_score(cs::ChoiceSite) = cs.score
getindex(cs::ChoiceSite, addr::Address) = nothing
haskey(cs::ChoiceSite, addr::Address) = false
unwrap(cs::ChoiceSite) = cs.val

# ------------ Site with a user-declared parameter ------------ #

struct ParameterSite{T} <: LearnableSite
    val::T
end

# ------------ includes - traces + call sites ------------ #

abstract type Trace end

include("traces/hierarchical.jl")
include("traces/vectorized.jl")
include("traces/cond.jl")

# ------------ Documentation ------------ #

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
mutable struct HierarchicalTrace <: Trace
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
mutable struct VectorizedTrace{C <: RecordSite} <: Trace
    subrecords::Vector{C}
    params::Dict{Address, LearnableSite}
end

```
Structured execution trace for `markov` and `plate` calls. The dependency structure interpretation of the `subrecords` vector depends on the call. For `markov`, the structure is Markovian. For `plate`, each element is drawn IID from the program or distribution provided to `plate`.
""", HierarchicalTrace)

@doc(
"""
```julia
mutable struct HierarchicalCallSite{J, K} <: CallSite
    trace::HierarchicalTrace
    score::Float64
    fn::Function
    args::J
    ret::K
end
```
A record of a black-box call (e.g. no special tracer language features). Records the `fn` and `args` for the call, as well as the `ret` return value.
""", HierarchicalCallSite)

@doc(
"""
```julia
mutable struct VectorizedCallSite{F <: Function, C <: RecordSite, J, K} <: CallSite
    trace::VectorizedTrace{C}
    score::Float64
    fn::Function
    args::J
    ret::Vector{K}
end
```
A record of a call site using the special `plate` and `markov` tracer language features. Informs the tracer that the call conforms to a special pattern of randomness dependency, which allows the storing of `Trace` instances sequentially in a vector.
""", VectorizedCallSite)
