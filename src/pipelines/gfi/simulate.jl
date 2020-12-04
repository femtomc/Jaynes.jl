# ------------ Choice sites ------------ #

@inline function (ctx::SimulateContext)(call::typeof(trace), 
                                        addr::T, 
                                        d::Distribution{K}) where {T <: Address, K}
    visit!(ctx.visited, addr)
    s = rand(d)
    add_choice!(ctx, addr, logpdf(d, s), s)
    return s
end

# ------------ Learnable ------------ #

@inline function (ctx::SimulateContext)(model::typeof(learnable), addr::T) where T <: Address
    visit!(ctx, addr)
    haskey(ctx.params, addr) && return getindex(ctx.params, addr)
    error("Parameter not provided at address $addr.")
end

# ------------ Black box call sites ------------ #

@inline function (ctx::SimulateContext)(c::typeof(trace),
                                        addr::T,
                                        call::Function,
                                        args...) where T <: Address
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ret, cl = simulate(ps, call, args...)
    add_call!(ctx, addr, cl)
    return ret
end

@inline function (ctx::SimulateContext)(c::typeof(trace),
                                        addr::T,
                                        call::G,
                                        args...) where {G <: GenerativeFunction,
                                                        T <: Address}
    visit!(ctx, addr)
    tr = simulate(call, args)
    ret = get_retval(tr)
    add_call!(ctx, addr, DynamicCallSite(get_choices(tr), get_score(tr), get_gen_fn(tr), get_args(tr), ret))
    ret
end

# ------------ Convenience ------------ #

function simulate(opt::J, params::P, model::Function, args...) where {J <: CompilationOptions, P <: AddressMap}
    ctx = Simulate(opt, DynamicTrace(), params)
    ret = ctx(model, args...)
    return ret, DynamicCallSite(ctx.tr, ctx.score, model, args, ret)
end

function simulate(params::P, model::Function, args...) where P <: AddressMap
    ctx = Simulate(DefaultPipeline(), DynamicTrace(), params)
    ret = ctx(model, args...)
    return ret, DynamicCallSite(ctx.tr, ctx.score, model, args, ret)
end

function simulate(model::Function, args...)
    ctx = Simulate(DefaultPipeline(), DynamicTrace(), Empty())
    ret = ctx(model, args...)
    return ret, DynamicCallSite(ctx.tr, ctx.score, model, args, ret)
end

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
