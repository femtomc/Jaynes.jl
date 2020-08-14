mutable struct SimulateContext{T <: AddressMap, 
                               P <: AddressMap} <: ExecutionContext
    tr::T
    score::Float64
    visited::Visitor
    params::P
end

function Simulate()
    SimulateContext(DynamicTrace(), 
                    0.0, 
                    Visitor(), 
                    Empty())
end

function Simulate(params)
    SimulateContext(DynamicTrace(), 
                    0.0, 
                    Visitor(), 
                    params)
end

# ------------ includes ------------ #

include("dynamic/simulate.jl")
include("plate/simulate.jl")
include("markov/simulate.jl")
include("conditional/simulate.jl")
include("factor/simulate.jl")

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct SimulateContext{T <: AddressMap, P <: AddressMap} <: ExecutionContext
    tr::T
    visited::Visitor
    params::P
    SimulateContext(params) where T <: AddressMap = new{T}(AddressMap(), Visitor(), params)
end
```

`SimulateContext` is used to simulate traces without recording likelihood weights. `SimulateContext` can be instantiated with custom `AddressMap` instances, which is useful when used for gradient-based learning.

Inner constructors:
```julia
SimulateContext(params) = new{DynamicAddressMap}(AddressMap(), Visitor(), params)
```
""", SimulateContext)

@doc(
"""
```julia
ret, cl = simulate(fn::Function, args...; params = LearnableByAddress())
ret, cl = simulate(fn::typeof(rand), d::Distribution{T}; params = LearnableByAddress()) where T
ret, v_cl = simulate(c::typeof(plate), fn::Function, args::Vector; params = LearnableByAddress()) where T
ret, v_cl = simulate(fn::typeof(plate), d::Distribution{T}, len::Int; params = LearnableByAddress()) where T
ret, v_cl = simulate(c::typeof(markov), fn::Function, len::Int, args...; params = LearnableByAddress())
```

`simulate` function provides an API to the `SimulateContext` execution context. You can use this function on any of the matching signatures above - it will return the return value `ret`, and a `RecordSite` instance specialized to the call. `simulate` is used to express unconstrained generation of a probabilistic program trace, without likelihood weight recording.
""", simulate)

