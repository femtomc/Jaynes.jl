# ------------ Propose compilation context ------------ #

mutable struct ProposeContext{J <: CompilationOptions,
                              T <: AddressMap, 
                              P <: AddressMap} <: ExecutionContext
    map::T
    score::Float64
    visited::Visitor
    params::P
    ProposeContext{J}(tr::T, score::Float64, visited::Visitor, params::P) where {J, T, P} = new{J, T, P}(tr, score, visited, params)
end

function Propose(opt::J, tr, params) where J
    ProposeContext{J}(tr, 
                      0.0, 
                      Visitor(), 
                      params)
end

# ------------ Dynamo ------------ #

@dynamo function (px::ProposeContext{J})(a...) where J
    ir = IR(a...)
    ir == nothing && return
    opt = extract_options(J)
    opt.AA == :on && jaynesize_transform!(ir)
    ir = recur(ir)
    ir
end

# Base fixes.
function (px::ProposeContext)(::typeof(Core._apply_iterate), f, c::typeof(trace), args...)
    flt = flatten(args)
    addr, rest = flt[1], flt[2 : end]
    ret, cl = propose(rest...)
    add_call!(px, addr, cl)
    ret
end

function (px::ProposeContext)(::typeof(Base.collect), generator::Base.Generator)
    map(generator.iter) do i
        px(generator.f, i)
    end
end

function propose(e::E, args...) where E <: ExecutionContext
    ctx = Propose(Trace(), Empty())
    ret = ctx(e, args...)
    ret, DynamicCallSite(ctx.tr, ctx.score, e, args, ret)
end

# ------------ Choice sites ------------ #

@inline function (ctx::ProposeContext)(call::typeof(trace), 
                                       addr::T, 
                                       d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)
    s = rand(d)
    score = logpdf(d, s)
    add_value!(ctx, addr, score, s)
    return s
end

# ------------ Learnable ------------ #

@inline function (ctx::ProposeContext)(fn::typeof(learnable), addr::Address)
    visit!(ctx, addr)
    has_sub(ctx.params, addr) && return get_sub(ctx.params, addr)
    error("Parameter not provided at address $addr.")
end

# ------------ Black box call sites ------------ #

@inline function (ctx::ProposeContext)(c::typeof(trace),
                                       addr::T,
                                       call::Function,
                                       args...) where T <: Address
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ret, submap, w = propose(ps, call, args...)
    set_sub!(ctx.map, addr, submap)
    return ret
end

@inline function (ctx::ProposeContext)(c::typeof(trace),
                                       addr::T,
                                       call::G,
                                       args...) where {G <: GenerativeFunction, T <: Address}
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ret, submap, w = propose(ps, call.fn, args...)
    set_sub!(ctx.map, addr, submap)
    return ret
end

# ------------ Convenience ------------ #

function propose(opt::J, params, fn::Function, args...) where J <: CompilationOptions
    ctx = Propose(opt, DynamicMap{Value}(), params)
    ret = ctx(fn, args...)
    return ret, ctx.map, ctx.score
end

function propose(params, fn::Function, args...) where J <: CompilationOptions
    ctx = Propose(DefaultPipeline(), DynamicMap{Value}(), params)
    ret = ctx(fn, args...)
    return ret, ctx.map, ctx.score
end

function propose(fn::Function, args...) where J <: CompilationOptions
    ctx = Propose(DefaultPipeline(), DynamicMap{Value}(), Empty())
    ret = ctx(fn, args...)
    return ret, ctx.map, ctx.score
end

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
