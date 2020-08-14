mutable struct ProposeContext{T <: AddressMap, P <: AddressMap} <: ExecutionContext
    tr::T
    score::Float64
    visited::Visitor
    params::P
    ProposeContext(tr::T) where T <: AddressMap = new{T, EmptyAddressMap}(tr, 0.0, Visitor(), AddressMap())
    ProposeContext(tr::T, params::P) where {T <: AddressMap, P} = new{T, P}(tr, 0.0, Visitor(), params)
end
Propose() = ProposeContext(AddressMap())
Propose(params) = ProposeContext(AddressMap(), params)

# ------------ includes ------------ #

include("dynamic/propose.jl")
include("plate/propose.jl")
include("markov/propose.jl")
include("factor/propose.jl")

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct ProposeContext{T <: AddressMap, P <: AddressMap} <: ExecutionContext
    tr::T
    score::Float64
    visited::Visitor
    params::P
end
```

`ProposeContext` is used to propose traces for inference algorithms which use custom proposals. `ProposeContext` instances can be passed sets of `AddressMap` to configure the propose with parameters which have been learned by differentiable programming.

Inner constructors:

```julia
ProposeContext(tr::T) where T <: AddressMap = new{T}(tr, 0.0, AddressMap())
```

Outer constructors:

```julia
Propose() = ProposeContext(AddressMap())
```
""", ProposeContext)

@doc(
"""
```julia
ret, g_cl, w = propose(fn::Function, args...)
ret, cs, w = propose(fn::typeof(rand), d::Distribution{K}) where K
ret, v_cl, w = propose(fn::typeof(markov), call::Function, len::Int, args...)
ret, v_cl, w = propose(fn::typeof(plate), call::Function, args::Vector)
ret, v_cl, w = propose(fn::typeof(plate), d::Distribution{K}, len::Int) where K
```

`propose` provides an API to the `ProposeContext` execution context. You can use this function on any of the matching signatures above - it will return the return value `ret`, a `RecordSite` instance specialized to the call, and the score `w`.
""", propose)
