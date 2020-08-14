mutable struct RegenerateContext{C <: AddressMap,
                                 T <: AddressMap, 
                                 K <: AddressMap,
                                 D <: AddressMap,
                                 P <: AddressMap,
                                 Ag <: Diff} <: ExecutionContext
    prev::C
    tr::T
    target::K
    weight::Float64
    score::Float64
    discard::D
    visited::Visitor
    params::P
    argdiffs::Ag
end
function Regenerate(target::K, cl::C) where {K <: AddressMap, C <: CallSite}
    RegenerateContext(cl, 
                  typeof(cl.trace)(), 
                  target, 
                  0.0, 
                  0.0, 
                  DynamicDiscard(), 
                  Visitor(), 
                  Empty(), 
                  NoChange())
end
function Regenerate(target::K, cl::C, argdiffs::Ag) where {K <: AddressMap, C <: CallSite, Ag <: Diff}
    RegenerateContext(cl, 
                  typeof(cl.trace)(), 
                  target, 
                  0.0, 
                  0.0, 
                  DynamicDiscard(), 
                  Visitor(), 
                  Empty(), 
                  argdiffs)
end

# ------------ includes ------------ #

include("dynamic/regenerate.jl")
#include("plate/regenerate.jl")
#include("markov/regenerate.jl")
#include("factor/regenerate.jl")

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct RegenerateContext{T <: Trace, 
                                 L <: Target,
                                 P <: AddressMap} <: ExecutionContext
    prev::T
    tr::T
    target::L
    weight::Float64
    score::Float64
    discard::T
    visited::Visitor
    params::P
end
```

Inner constructors:

```julia
function RegenerateContext(tr::T, sel::Vector{Address}) where T <: Trace
    un_sel = targetion(sel)
    new{T, typeof(un_sel), EmptyAddressMap}(tr, Trace(), un_sel, 0.0, Trace(), Visitor(), AddressMap())
end
function RegenerateContext(tr::T, sel::L) where {T <: Trace, L <: Target}
    new{T, L, EmptyAddressMap}(tr, Trace(), sel, 0.0, Trace(), Visitor(), AddressMap())
end
```

Outer constructors:

```julia
Regenerate(tr::Trace, sel::Vector{Address}) = RegenerateContext(tr, sel)
Regenerate(tr::Trace, sel::Target) = RegenerateContext(tr, sel)
```

The `RegenerateContext` is used for MCMC algorithms, to propose new choices for addresses indicated by an `Target` in the `target` field.
""", RegenerateContext)

@doc(
"""
```julia
ret, cl = regenerate(sel::L, cs::DynamicCallSite, new_args...) where L <: Target
ret, cl = regenerate(sel::L, cs::DynamicCallSite) where L <: Target
```
`regenerate` is an API to the `RegenerateContext` execution context. `regenerate` requires that users provide an `Target`, an original call site, and possibly a set of new arguments to be used in the regeneration step. This context internally keeps track of the bookkeeping required to increment likelihood weights, as well as prune off parts of the trace which are invalid if a regenerated choice changes the shape of the trace (e.g. control flow), and returns a new return value `ret` as well as the modified call site `cl`.
""", regenerate)
