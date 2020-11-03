mutable struct ScoreContext{M <: AddressMap,
                            P <: AddressMap} <: ExecutionContext
    target::M
    weight::Float64
    visited::Visitor
    params::P
end

function Score(obs::Vector)
    ScoreContext(target(obs),
                 0.0,
                 Visitor(),
                 Empty())
end

function Score(obs::AddressMap)
    ScoreContext(obs, 
                 0.0, 
                 Visitor(), 
                 Empty())
end

function Score(obs::AddressMap, params)
    ScoreContext(obs, 
                 0.0, 
                 Visitor(),
                 params)
end

@dynamo function (sx::ScoreContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    jaynesize_transform!(ir)
    ir = recur(ir)
    ir
end
(sx::ScoreContext)(::typeof(Core._apply_iterate), f, c::typeof(trace), args...) = sx(c, flatten(args)...)
function (sx::ScoreContext)(::typeof(Base.collect), generator::Base.Generator)
    map(generator.iter) do i
        sx(generator.f, i)
    end
end

# ------------ includes ------------ #

include("dynamic/score.jl")
include("factor/score.jl")

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct ScoreContext{P <: AddressMap} <: ExecutionContext
    select::AddressMap
    weight::Float64
    params::P
end
```

The `ScoreContext` is used to score selections according to a model function. For computation in the `ScoreContext` to execute successfully, the `select` selection must provide a choice for every address visited in the model function, and the model function must allow the context to visit every constraints expressed in `select`.

Inner constructors:

```julia
function Score(obs::Vector{Tuple{K, P}}) where {P, K <: Union{Symbol, Pair}}
    c_sel = selection(obs)
    new{EmptyAddressMap}(c_sel, 0.0, AddressMap())
end
```

Outer constructors:

```julia
ScoreContext(obs::K, params) where {K <: AddressMap} = new(obs, 0.0, params)
Score(obs::Vector) = ScoreContext(selection(obs))
Score(obs::AddressMap) = ScoreContext(obs, AddressMap())
Score(obs::AddressMap, params) = ScoreContext(obs, params)
```
""", ScoreContext)

@doc(
"""
```julia
ret, w = score(sel::L, fn::Function, args...; params = AddressMap()) where L <: AddressMap
ret, w = score(sel::L, fn::typeof(rand), d::Distribution{K}; params = AddressMap()) where {L <: AddressMap, K}
```

`score` provides an API to the `ScoreContext` execution context. You can use this function on any of the matching signatures above - it will return the return value `ret`, and the likelihood weight score of the user-provided selection `sel`. The selection should satisfy the following requirement:

1. At any random choice in any branch traveled according to the constraints of `sel`, `sel` must provide a constraint for that choice.

Simply put, this just means you need to provide a constraint for each `ChoiceSite` you encounter.
""", score)
