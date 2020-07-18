mutable struct SimulateContext{T <: Trace} <: ExecutionContext
    tr::T
    score::Float64
    visited::Visitor
    params::LearnableParameters
    SimulateContext(params) = new{HierarchicalTrace}(Trace(), 0.0, Visitor(), params)
end

# ------------ Convenience ------------ #

function simulate(fn::Function, args...; params = LearnableParameters())
    ctx = SimulateContext(params)
    ret = ctx(fn, args...)
    return ret, HierarchicalCallSite(ctx.tr, ctx.score, fn, args, ret)
end

function simulate(fn::typeof(rand), d::Distribution{T}; params = LearnableParameters()) where T
    ctx = SimulateContext(params)
    addr = gensym()
    ret = ctx(rand, addr, d)
    return ret, get_choice(ctx.tr, addr)
end

function simulate(c::typeof(plate), fn::Function, args::Vector; params = LearnableParameters()) where T
    ctx = SimulateContext(params)
    addr = gensym()
    ret = ctx(plate, addr, fn, args)
    return ret, get_call(ctx.tr, addr)
end

function simulate(fn::typeof(plate), d::Distribution{T}, len::Int; params = LearnableParameters()) where T
    ctx = SimulateContext(params)
    addr = gensym()
    ret = ctx(plate, addr, d, len)
    return ret, get_call(ctx.tr, addr)
end

#function simulate(::Tuple{typeof(plate), N}, d::Distribution{T}, len::Int; params = LearnableParameters()) where T
#    ctx = SimulateContext(params)
#    addr = gensym()
#    ret = ctx(plate, addr, d, len)
#    return ret, get_call(ctx.tr, addr)
#end

function simulate(c::typeof(markov), fn::Function, len::Int, args...; params = LearnableParameters())
    ctx = SimulateContext(params)
    addr = gensym()
    ret = ctx(markov, addr, fn, len, args...)
    return ret, get_call(ctx.tr, addr)
end

# ------------ includes ------------ #

include("hierarchical/simulate.jl")
include("plate/simulate.jl")
include("markov/simulate.jl")
include("cond/simulate.jl")

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
simulate(fn::Function, args...; params = LearnableParameters())
simulate(fn::typeof(rand), d::Distribution{T}; params = LearnableParameters()) where T
simulate(c::typeof(plate), fn::Function, args::Vector; params = LearnableParameters()) where T
simulate(fn::typeof(plate), d::Distribution{T}, len::Int; params = LearnableParameters()) where T
simulate(c::typeof(markov), fn::Function, len::Int, args...; params = LearnableParameters())
```

The convenience `simulate` function provides an API to the `Simulate` context. You can use this function on any of the matching signatures above - it will return the return value `ret`, and a `RecordSite` instance specialized to the call. `simulate` is used to express unconstrained generation of a probabilistic program trace, without likelihood weight recording.
""", simulate)

