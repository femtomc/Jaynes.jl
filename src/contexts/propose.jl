mutable struct ProposeContext{T <: Trace, P <: Parameters} <: ExecutionContext
    tr::T
    score::Float64
    visited::Visitor
    params::P
    ProposeContext(tr::T) where T <: Trace = new{T, EmptyParameters}(tr, 0.0, Visitor(), Parameters())
    ProposeContext(tr::T, params::P) where {T <: Trace, P} = new{T, P}(tr, 0.0, Visitor(), params)
end
Propose() = ProposeContext(Trace())
Propose(params) = ProposeContext(Trace(), params)

# ------------ Convenience ------------ #

function propose(fn::Function, args...)
    ctx = Propose()
    ret = ctx(fn, args...)
    return ret, HierarchicalCallSite(ctx.tr, ctx.score, fn, args, ret), ctx.score
end

function propose(params, fn::Function, args...)
    ctx = Propose(params)
    ret = ctx(fn, args...)
    return ret, HierarchicalCallSite(ctx.tr, ctx.score, fn, args, ret), ctx.score
end

function propose(fn::typeof(rand), d::Distribution{K}) where K
    ctx = Propose()
    addr = gensym()
    ret = ctx(fn, addr, d)
    return ret, get_top(ctx.tr, addr), ctx.score
end

function propose(fn::typeof(markov), call::Function, len::Int, args...)
    ctx = Propose()
    addr = gensym()
    ret = ctx(fn, addr, call, len, args...)
    return ret, get_sub(ctx.tr, addr), ctx.score
end

function propose(params, fn::typeof(markov), call::Function, len::Int, args...)
    ctx = Propose(params)
    addr = gensym()
    ret = ctx(fn, addr, call, len, args...)
    return ret, get_sub(ctx.tr, addr), ctx.score
end

function propose(fn::typeof(plate), call::Function, args::Vector)
    ctx = Propose(params)
    addr = gensym()
    ret = ctx(fn, addr, call, args)
    return ret, get_sub(ctx.tr, addr), ctx.score
end

function propose(params, fn::typeof(plate), call::Function, args::Vector)
    ctx = Propose(params)
    addr = gensym()
    ret = ctx(fn, addr, call, args)
    return ret, get_sub(ctx.tr, addr), ctx.score
end

function propose(fn::typeof(plate), d::Distribution{K}, len::Int) where K
    ctx = Propose()
    addr = gensym()
    ret = ctx(fn, addr, d, len)
    return ret, get_sub(ctx.tr, addr), ctx.score
end

# ------------ includes ------------ #

include("hierarchical/propose.jl")
include("plate/propose.jl")
include("markov/propose.jl")
include("factor/propose.jl")

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct ProposeContext{T <: Trace, P <: Parameters} <: ExecutionContext
    tr::T
    score::Float64
    visited::Visitor
    params::P
end
```

`ProposeContext` is used to propose traces for inference algorithms which use custom proposals. `ProposeContext` instances can be passed sets of `Parameters` to configure the propose with parameters which have been learned by differentiable programming.

Inner constructors:

```julia
ProposeContext(tr::T) where T <: Trace = new{T}(tr, 0.0, Parameters())
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

`propose` provides an API to the `ProposeContext` execution context. You can use this function on any of the matching signatures above - it will return the return value `ret`, a `RecordSite` instance specialized to the call, and the score `w`.
""", propose)
