mutable struct SimulateContext{T <: Trace, P <: Parameters} <: ExecutionContext
    tr::T
    score::Float64
    visited::Visitor
    params::P
    SimulateContext() = new{HierarchicalTrace, EmptyParameters}(Trace(), 0.0, Visitor(), Parameters())
    SimulateContext(params::P) where P = new{HierarchicalTrace, P}(Trace(), 0.0, Visitor(), params)
end
Simulate() = SimulateContext()

# ------------ Convenience ------------ #

function simulate(fn::Function, args...)
    ctx = SimulateContext()
    ret = ctx(fn, args...)
    return ret, HierarchicalCallSite(ctx.tr, ctx.score, fn, args, ret)
end

function simulate(params::P, fn::Function, args...) where P <: Parameters
    ctx = SimulateContext(params)
    ret = ctx(fn, args...)
    return ret, HierarchicalCallSite(ctx.tr, ctx.score, fn, args, ret)
end

function simulate(fn::typeof(rand), d::Distribution{T}) where T
    ctx = SimulateContext()
    addr = gensym()
    ret = ctx(rand, addr, d)
    return ret, get_top(ctx.tr, addr)
end

function simulate(params::P, fn::typeof(rand), d::Distribution{T}) where {P <: Parameters, T}
    ctx = SimulateContext(params)
    addr = gensym()
    ret = ctx(rand, addr, d)
    return ret, get_top(ctx.tr, addr)
end

function simulate(c::typeof(plate), fn::Function, args::Vector)
    ctx = SimulateContext()
    addr = gensym()
    ret = ctx(plate, addr, fn, args)
    return ret, get_sub(ctx.tr, addr)
end

function simulate(params::P, c::typeof(plate), fn::Function, args::Vector) where P <: Parameters
    addr = gensym()
    v_ps = learnables(addr => params)
    ctx = SimulateContext(v_ps)
    ret = ctx(plate, addr, fn, args)
    return ret, get_sub(ctx.tr, addr)
end

function simulate(fn::typeof(plate), d::Distribution{T}, len::Int) where T
    ctx = SimulateContext()
    addr = gensym()
    ret = ctx(plate, addr, d, len)
    return ret, get_sub(ctx.tr, addr)
end

function simulate(c::typeof(markov), fn::Function, len::Int, args...)
    ctx = SimulateContext()
    addr = gensym()
    ret = ctx(markov, addr, fn, len, args...)
    return ret, get_sub(ctx.tr, addr)
end

function simulate(params::P, c::typeof(markov), fn::Function, len::Int, args...) where P <: Parameters
    addr = gensym()
    v_ps = learnables(addr => params)
    ctx = SimulateContext(v_ps)
    ret = ctx(markov, addr, fn, len, args...)
    return ret, get_sub(ctx.tr, addr)
end

# ------------ includes ------------ #

include("hierarchical/simulate.jl")
include("plate/simulate.jl")
include("markov/simulate.jl")
include("conditional/simulate.jl")
include("factor/simulate.jl")

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct SimulateContext{T <: Trace, P <: Parameters} <: ExecutionContext
    tr::T
    visited::Visitor
    params::P
    SimulateContext(params) where T <: Trace = new{T}(Trace(), Visitor(), params)
end
```

`SimulateContext` is used to simulate traces without recording likelihood weights. `SimulateContext` can be instantiated with custom `Parameters` instances, which is useful when used for gradient-based learning.

Inner constructors:
```julia
SimulateContext(params) = new{HierarchicalTrace}(Trace(), Visitor(), params)
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

