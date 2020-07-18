mutable struct ProposeContext{T <: Trace} <: ExecutionContext
    tr::T
    weight::Float64
    score::Float64
    visited::Visitor
    params::LearnableParameters
    ProposeContext(tr::T) where T <: Trace = new{T}(tr, 0.0, 0.0, Visitor(), LearnableParameters())
    ProposeContext(tr::T, params::LearnableParameters) where T <: Trace = new{T}(tr, 0.0, 0.0, Visitor(), params)
end
Propose() = ProposeContext(Trace())
Propose(params) = ProposeContext(Trace(), params)

# ------------ Convenience ------------ #

function propose(fn::Function, args...; params = LearnableParameters())
    ctx = Propose(params)
    ret = ctx(fn, args...)
    return ret, HierarchicalCallSite(ctx.tr, ctx.score, fn, args, ret), ctx.weight
end

function propose(fn::typeof(rand), d::Distribution{K}; params = LearnableParameters()) where K
    ctx = Propose(params)
    addr = gensym()
    ret = ctx(fn, addr, d)
    return ret, get_choice(ctx.tr, addr), ctx.weight
end

function propose(fn::typeof(markov), call::Function, len::Int, args...; params = LearnableParameters())
    ctx = Propose(params)
    addr = gensym()
    ret = ctx(fn, addr, call, len, args...)
    return ret, get_call(ctx.tr, addr), ctx.weight
end

function propose(fn::typeof(plate), call::Function, args::Vector; params = LearnableParameters())
    ctx = Propose(params)
    addr = gensym()
    ret = ctx(fn, addr, call, args)
    return ret, get_call(ctx.tr, addr), ctx.weight
end

function propose(fn::typeof(plate), d::Distribution{K}, len::Int; params = LearnableParameters()) where K
    ctx = Propose(params)
    addr = gensym()
    ret = ctx(fn, addr, d, len)
    return ret, get_call(ctx.tr, addr), ctx.weight
end

# ------------ includes ------------ #

include("hierarchical/propose.jl")
include("plate/propose.jl")
include("markov/propose.jl")

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct ProposeContext{T <: Trace} <: ExecutionContext
    tr::T
    weight::Float64
    score::Float64
    visited::Visitor
    params::LearnableParameters
end
```

`ProposeContext` is used to propose traces for inference algorithms which use custom proposals. `ProposeContext` instances can be passed sets of `LearnableParameters` to configure the propose with parameters which have been learned by differentiable programming.

Inner constructors:

```julia
ProposeContext(tr::T) where T <: Trace = new{T}(tr, 0.0, LearnableParameters())
```

Outer constructors:

```julia
Propose() = ProposeContext(Trace())
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

`propose` provides an API to the `ProposeContext` execution context. You can use this function on any of the matching signatures above - it will return the return value `ret`, a `RecordSite` instance specialized to the call, and the score/weight `w`.
""", propose)
