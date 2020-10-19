mutable struct GenerateContext{T <: AddressMap, 
                               K <: AddressMap, 
                               P <: AddressMap} <: ExecutionContext
    tr::T
    target::K
    weight::Float64
    score::Float64
    visited::Visitor
    params::P
end

function Generate(target::AddressMap)
    GenerateContext(DynamicTrace(), 
                    target, 
                    0.0,
                    0.0,
                    Visitor(),
                    Empty())
end

function Generate(target::AddressMap, params)
    GenerateContext(DynamicTrace(), 
                    target, 
                    0.0,
                    0.0,
                    Visitor(),
                    params)
end

function Generate(tr::AddressMap, target::AddressMap, params::AddressMap)
    GenerateContext(tr, 
                    target,
                    0.0,
                    0.0,
                    Visitor(),
                    params)
end

# Go go dynamo!
@dynamo function (gx::GenerateContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    transform!(ir)
    ir = recur(ir)
    ir
end
(gx::GenerateContext)(::typeof(Core._apply_iterate), f, c::typeof(trace), args...) = gx(c, flatten(args)...)
function (gx::GenerateContext)(::typeof(Base.collect), generator::Base.Generator)
    map(generator.iter) do i
        gx(generator.f, i)
    end
end

# ------------ includes ------------ #

include("dynamic/generate.jl")
include("factor/generate.jl")

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct GenerateContext{T <: AddressMap, K <: AddressMap, P <: Parameters} <: ExecutionContext
     tr::T
     target::K
     weight::Float64
     score::Float64
     visited::Visitor
     params::P
end
```
`GenerateContext` is used to generate traces, as well as record and accumulate likelihood weights given observations at addressed randomness.

Inner constructors:
```julia
GenerateContext(tr::T, target::K) where {T <: AddressMap, K <: AddressMap} = new{T, K}(tr, target, 0.0, Visitor(), Parameters())
GenerateContext(tr::T, target::K, params::P) where {T <: AddressMap, K <: AddressMap, P <: Parameters} = new{T, K, P}(tr, target, 0.0, Visitor(), params)
```
Outer constructors:
```julia
Generate(target::AddressMap) = GenerateContext(AddressMap(), target)
Generate(target::AddressMap, params) = GenerateContext(AddressMap(), target, params)
Generate(tr::AddressMap, target::AddressMap) = GenerateContext(tr, target)
```
""", GenerateContext)

@doc(
"""
```julia
ret, cl, w = generate(target::L, fn::Function, args...; params = Parameters()) where L <: AddressMap
ret, cs, w = generate(target::L, fn::typeof(rand), d::Distribution{K}; params = Parameters()) where {L <: AddressMap, K}
ret, v_cl, w = generate(target::L, fn::typeof(markov), call::Function, len::Int, args...; params = Parameters()) where L <: AddressMap
ret, v_cl, w = generate(target::L, fn::typeof(plate), call::Function, args::Vector; params = Parameters()) where L <: AddressMap
ret, v_cl, w = generate(target::L, fn::typeof(plate), d::Distribution{K}, len::Int; params = Parameters()) where {L <: AddressMap, K}
```

`generate` provides an API to the `GenerateContext` execution context. You can use this function on any of the matching signatures above - it will return the return value `ret`, a `RecordSite` instance specialized to the call, and the score/weight `w` computed with respect to the constraints `target`.
""", generate)

