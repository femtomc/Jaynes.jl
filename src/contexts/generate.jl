mutable struct GenerateContext{T <: AddressMap, K <: AddressMap, P <: AddressMap} <: ExecutionContext
    tr::T
    schema::K
    weight::Float64
    score::Float64
    visited::Visitor
    params::P
    GenerateContext(tr::T, schema::K) where {T <: AddressMap, K <: AddressMap} = new{T, K, Empty}(tr, schema, 0.0, 0.0, Visitor(), Empty())
    GenerateContext(tr::T, schema::K, params::P) where {T <: AddressMap, K <: AddressMap, P <: AddressMap} = new{T, K, P}(tr, schema, 0.0, 0.0, Visitor(), params)
end
Generate(schema::AddressMap) = GenerateContext(Trace(), schema, Parameters())
Generate(schema::AddressMap, params) = GenerateContext(Trace(), schema, params)
Generate(tr::AddressMap, schema::AddressMap) = GenerateContext(tr, schema)

# ------------ includes ------------ #

include("dynamic/generate.jl")
include("plate/generate.jl")
include("markov/generate.jl")
include("conditional/generate.jl")
include("factor/generate.jl")

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct GenerateContext{T <: AddressMap, K <: AddressMap, P <: Parameters} <: ExecutionContext
     tr::T
     schema::K
     weight::Float64
     score::Float64
     visited::Visitor
     params::P
end
```
`GenerateContext` is used to generate traces, as well as record and accumulate likelihood weights given observations at addressed randomness.

Inner constructors:
```julia
GenerateContext(tr::T, schema::K) where {T <: AddressMap, K <: AddressMap} = new{T, K}(tr, schema, 0.0, Visitor(), Parameters())
GenerateContext(tr::T, schema::K, params::P) where {T <: AddressMap, K <: AddressMap, P <: Parameters} = new{T, K, P}(tr, schema, 0.0, Visitor(), params)
```
Outer constructors:
```julia
Generate(schema::AddressMap) = GenerateContext(AddressMap(), schema)
Generate(schema::AddressMap, params) = GenerateContext(AddressMap(), schema, params)
Generate(tr::AddressMap, schema::AddressMap) = GenerateContext(tr, schema)
```
""", GenerateContext)

@doc(
"""
```julia
ret, cl, w = generate(schema::L, fn::Function, args...; params = Parameters()) where L <: AddressMap
ret, cs, w = generate(schema::L, fn::typeof(rand), d::Distribution{K}; params = Parameters()) where {L <: AddressMap, K}
ret, v_cl, w = generate(schema::L, fn::typeof(markov), call::Function, len::Int, args...; params = Parameters()) where L <: AddressMap
ret, v_cl, w = generate(schema::L, fn::typeof(plate), call::Function, args::Vector; params = Parameters()) where L <: AddressMap
ret, v_cl, w = generate(schema::L, fn::typeof(plate), d::Distribution{K}, len::Int; params = Parameters()) where {L <: AddressMap, K}
```

`generate` provides an API to the `GenerateContext` execution context. You can use this function on any of the matching signatures above - it will return the return value `ret`, a `RecordSite` instance specialized to the call, and the score/weight `w` computed with respect to the constraints `schema`.
""", generate)

