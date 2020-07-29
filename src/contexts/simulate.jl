mutable struct SimulateContext{T <: Trace, P <: Parameters} <: ExecutionContext
    tr::T
    score::Float64
    visited::Visitor
    params::P
    SimulateContext() = new{HierarchicalTrace, EmptyParameters}(Trace(), 0.0, Visitor(), Parameters())
    SimulateContext(params::P) where P = new{HierarchicalTrace, P}(Trace(), 0.0, Visitor(), params)
end

# ------------ Convenience ------------ #

function simulate(fn::Function, args...)
    ctx = SimulateContext()
    ret = ctx(fn, args...)
    return ret, HierarchicalCallSite(ctx.tr, ctx.score, fn, args, ret)
end

function simulate(params, fn::Function, args...)
    ctx = SimulateContext(params)
    ret = ctx(fn, args...)
    return ret, HierarchicalCallSite(ctx.tr, ctx.score, fn, args, ret)
end

function simulate(fn::typeof(rand), d::Distribution{T}) where T
    ctx = SimulateContext()
    addr = gensym()
    ret = ctx(rand, addr, d)
    return ret, get_choice(ctx.tr, addr)
end

function simulate(params, fn::typeof(rand), d::Distribution{T}) where T
    ctx = SimulateContext(params)
    addr = gensym()
    ret = ctx(rand, addr, d)
    return ret, get_choice(ctx.tr, addr)
end

function simulate(c::typeof(plate), fn::Function, args::Vector)
    ctx = SimulateContext()
    addr = gensym()
    ret = ctx(plate, addr, fn, args)
    return ret, get_call(ctx.tr, addr)
end

function simulate(params, c::typeof(plate), fn::Function, args::Vector)
    ctx = SimulateContext(params)
    addr = gensym()
    ret = ctx(plate, addr, fn, args)
    return ret, get_call(ctx.tr, addr)
end

function simulate(fn::typeof(plate), d::Distribution{T}, len::Int) where T
    ctx = SimulateContext()
    addr = gensym()
    ret = ctx(plate, addr, d, len)
    return ret, get_call(ctx.tr, addr)
end

function simulate(c::typeof(markov), fn::Function, len::Int, args...)
    ctx = SimulateContext()
    addr = gensym()
    ret = ctx(markov, addr, fn, len, args...)
    return ret, get_call(ctx.tr, addr)
end

function simulate(params, c::typeof(markov), fn::Function, len::Int, args...)
    ctx = SimulateContext(params)
    addr = gensym()
    ret = ctx(markov, addr, fn, len, args...)
    return ret, get_call(ctx.tr, addr)
end

# ------------ includes ------------ #

include("hierarchical/simulate.jl")
include("plate/simulate.jl")
include("markov/simulate.jl")
include("conditional/simulate.jl")

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct SimulateContext{T <: Trace} <: ExecutionContext
    tr::T
    visited::Visitor
    params::LearnableParameters
    SimulateContext(params) where T <: Trace = new{T}(Trace(), Visitor(), params)
end
```

`SimulateContext` is used to simulate traces without recording likelihood weights. `SimulateContext` can be instantiated with custom `LearnableParameters` instances, which is useful when used for gradient-based learning.

Inner constructors:
```julia
SimulateContext(params) = new{HierarchicalTrace}(Trace(), Visitor(), params)
```
""", SimulateContext)

@doc(
"""
```julia
ret, cl = simulate(fn::Function, args...; params = LearnableParameters())
ret, cl = simulate(fn::typeof(rand), d::Distribution{T}; params = LearnableParameters()) where T
ret, v_cl = simulate(c::typeof(plate), fn::Function, args::Vector; params = LearnableParameters()) where T
ret, v_cl = simulate(fn::typeof(plate), d::Distribution{T}, len::Int; params = LearnableParameters()) where T
ret, v_cl = simulate(c::typeof(markov), fn::Function, len::Int, args...; params = LearnableParameters())
```

`simulate` function provides an API to the `SimulateContext` execution context. You can use this function on any of the matching signatures above - it will return the return value `ret`, and a `RecordSite` instance specialized to the call. `simulate` is used to express unconstrained generation of a probabilistic program trace, without likelihood weight recording.
""", simulate)

