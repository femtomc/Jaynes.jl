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

include("generic/simulate.jl")
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
