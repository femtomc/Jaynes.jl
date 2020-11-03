mutable struct ProposeContext{T <: AddressMap, 
                              P <: AddressMap} <: ExecutionContext
    map::T
    score::Float64
    visited::Visitor
    params::P
end

function Propose(tr)
    ProposeContext(tr, 
                   0.0, 
                   Visitor(), 
                   Empty())
end

function Propose(tr, params)
    ProposeContext(tr, 
                   0.0, 
                   Visitor(), 
                   params)
end

# Go go dynamo!
@dynamo function (px::ProposeContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    jaynesize_transform!(ir)
    ir = recur(ir)
    ir
end
(px::ProposeContext)(::typeof(Core._apply_iterate), f, c::typeof(trace), args...) = px(c, flatten(args)...)
function (px::ProposeContext)(::typeof(Base.collect), generator::Base.Generator)
    map(generator.iter) do i
        px(generator.f, i)
    end
end

# ------------ includes ------------ #

include("dynamic/propose.jl")
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
```

`propose` provides an API to the `ProposeContext` execution context. You can use this function on any of the matching signatures above - it will return the return value `ret`, a `RecordSite` instance specialized to the call, and the score `w`.
""", propose)
