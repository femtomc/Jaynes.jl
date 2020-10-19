mutable struct SimulateContext{T <: AddressMap, 
                               P <: AddressMap} <: ExecutionContext
    tr::T
    score::Float64
    visited::Visitor
    params::P
end

function Simulate(tr, params)
    SimulateContext(tr,
                    0.0, 
                    Visitor(), 
                    params)
end

@dynamo function (sx::SimulateContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    transform!(ir)
    ir = recur(ir)
    ir
end
(sx::SimulateContext)(::typeof(Core._apply_iterate), f, c::typeof(trace), args...) = sx(c, flatten(args)...)
function (sx::SimulateContext)(::typeof(Base.collect), generator::Base.Generator)
    map(generator.iter) do i
        sx(generator.f, i)
    end
end

# ------------ includes ------------ #

include("dynamic/simulate.jl")
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
ret, cl = simulate(fn::typeof(trace), d::Distribution{T}; params = LearnableByAddress()) where T
```

`simulate` function provides an API to the `SimulateContext` execution context. You can use this function on any of the matching signatures above - it will return the return value `ret`, and a `RecordSite` instance specialized to the call. `simulate` is used to express unconstrained generation of a probabilistic program trace, without likelihood weight recording.
""", simulate)

