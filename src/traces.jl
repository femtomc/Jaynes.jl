import Base: getindex, haskey, rand, iterate

# ------------ Core ------------ #

Base.iterate(s::Symbol) = s

# Special calls recognized by tracer.
rand(addr::A, d::Distribution{T}) where {A <: Address, T} = error("(rand) call with address $addr evaluated outside of the tracer.\nThis normally occurs because you're not matching dispatch correctly, or you've forgotten to tell the tracer to recurse into a call site (wrap it with rand).")
rand(addr::A, fn::Function, args...) where A <: Address = error("(rand) call with address $addr evaluated outside of the tracer.\nThis normally occurs because you're not matching dispatch correctly, or you've forgotten to tell the tracer to recurse into a call site (wrap it with rand).")
rand(addr::A, fn::Function, args::Tuple, ret_score::Function) where A <: Address = error("(rand) call with address $addr evaluated outside of the tracer.\nThis normally occurs because you're not matching dispatch correctly, or you've forgotten to tell the tracer to recurse into a call site (wrap it with rand).")
learnable(addr::A) where {A <: Address} = error("(learnable) call with address $addr evaluated outside of the tracer.\nThis normally occurs because you're not matching dispatch correctly.")
plate(addr::A, args...) where A <: Address = error("(plate) call with address $addr evaluated outside of the tracer.\nThis normally occurs because you're not matching the dispatch correctly.")
markov(addr::A, args...) where A <: Address = error("(markov) call with address $addr evaluated outside of the tracer.\nThis normally occurs because you're not matching the dispatch correctly.")
cond(addr::A, args...) where A <: Address = error("(cond) call with address $addr evaluated outside of the tracer.\nThis normally occurs because you're not matching the dispatch correctly.")
gen_fmi(addr::A, args...) where A <: Address = error("(gen_fmi) call with address $addr evaluated outside of the tracer.\nThis normally occurs because you're not matching the dispatch correctly.")
soss_fmi(addr::A, args...) where A <: Address = error("(soss_fmi) call with address $addr evaluated outside of the tracer.\nThis normally occurs because you're not matching the dispatch correctly.")

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
get_ret(cs::ChoiceSite) = cs.val

# ------------ Site with a user-declared parameter ------------ #

struct ParameterSite{T} <: LearnableSite
    val::T
end

# ------------ includes - traces + call sites ------------ #

abstract type Trace end

include("traces/hierarchical.jl")
include("traces/vectorized.jl")
include("traces/conditional.jl")

# ------------ Pretty printing ------------ #

function Base.display(call::C; 
                      fields::Array{Symbol, 1} = [:val],
                      show_full = false) where C <: CallSite
    println("  __________________________________\n")
    println("               Playback\n")
    println(" type : $C\n")
    map(fieldnames(C)) do f
        val = getfield(call, f)
        typeof(val) <: Dict{Address, ChoiceSite} && begin 
            vals = collect(val)
            if length(vals) > 5 && !show_full
                map(vals[1:5]) do (k, v)
                    println(" $(k)")
                    map(fieldnames(ChoiceSite)) do nm
                        !(nm in fields) && return
                        println("          $(nm)  = $(getfield(v, nm))")
                    end
                    println("")
                end
                println("                  ...\n")
                println("  __________________________________\n")
                return
            else
                map(vals) do (k, v)
                    println(" $(k)")
                    map(fieldnames(ChoiceSite)) do nm
                        !(nm in fields) && return
                        println("          $(nm)  = $(getfield(v, nm))")
                    end
                    println("")
                end
                println("  __________________________________\n")
                return
            end
        end
        typeof(val) <: Real && begin
            println(" $(f) : $(val)\n")
            return
        end
        println(" $(f) : $(typeof(val))\n")
    end
    println("  __________________________________\n")
end

import Base.collect
function collect(tr::Trace)
    addrs = Any[]
    chd = Dict{Any, Any}()
    meta = Dict()
    collect!(addrs, chd, tr, meta)
    return addrs, chd, meta
end

function Base.display(tr::Trace; 
                      show_values = true, 
                      show_types = false)
    println("  __________________________________\n")
    println("               Addresses\n")
    addrs, chd, meta = collect(tr)
    if show_values
        for a in addrs
            if haskey(meta, a)
                println("$(meta[a]) $(a) = $(chd[a])")
            else
                println("$(a) = $(chd[a])")
            end
        end
    elseif show_types
        for a in addrs
            println(" $(a) = $(typeof(chd[a]))")
        end
    elseif show_types && show_values
        for a in addrs
            println(" $(a) = $(chd[a]) : $(typeof(chd[a]))")
        end
    else
        for a in addrs
            println(" $(a)")
        end
    end
    println("  __________________________________\n")
end

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
A record of a black-box call (e.g. no special tracer language features). Records the `fn` and `args` for the call, as well as the `ret` return value.
""", HierarchicalCallSite)

@doc(
"""
```julia
struct VectorizedCallSite{F <: Function, C <: RecordSite, J, K} <: CallSite
    trace::VectorizedTrace{C}
    score::Float64
    fn::Function
    args::J
    ret::Vector{K}
end
```
A record of a call site using the special `plate` and `markov` tracer language features. Informs the tracer that the call conforms to a special pattern of randomness dependency, which allows the storing of `Trace` instances sequentially in a vector.
""", VectorizedCallSite)
