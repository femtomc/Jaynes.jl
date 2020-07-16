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

# ------------ Choice sites ------------ #

@inline function (ctx::ProposeContext)(call::typeof(rand), 
                                       addr::T, 
                                       d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)
    s = rand(d)
    score = logpdf(d, s)
    add_choice!(ctx, addr, ChoiceSite(score, s))
    increment!(ctx, score)
    return s
end

@inline function (ctx::ProposeContext)(c::typeof(plate), 
                                        addr::T, 
                                        d::Distribution{K},
                                        len::Int) where {T <: Address, K}
    v_ret = Vector{eltype(d)}(undef, len)
    v_cs = Vector{ChoiceSite{eltype(d)}}(undef, len)
    for i in 1:len
        visit!(ctx, addr => i)
        s = rand(d)
        score = logpdf(d, s)
        cs = ChoiceSite(score, s)
        v_ret[i] = s
        v_cs[i] = cs
        increment!(ctx, score)
    end
    sc = sum(map(v_cs) do cs
                 get_score(cs)
             end)
    add_call!(ctx, addr, VectorizedSite{typeof(markov)}(VectorizedTrace(v_cs), sc, d, (), v_ret))
    return v_ret
end

# ------------ Learnable ------------ #

@inline function (ctx::ProposeContext)(fn::typeof(learnable), addr::Address, p::T) where T
    visit!(ctx, addr)
    ret = p
    if has_param(ctx.params, addr)
        ret = get_param(ctx.params, addr)
    end
    return ret
end

# ------------ Black box call sites ------------ #

@inline function (ctx::ProposeContext)(c::typeof(rand),
                                        addr::T,
                                        call::Function,
                                        args...) where T <: Address
    visit!(ctx, addr)
    ret, cl, w = propose(call, args...)
    add_call!(ctx, addr, cl)
    increment!(ctx, w)
    return ret
end

# ------------ Vectorized call sites ------------ #

@inline function (ctx::ProposeContext)(c::typeof(markov), 
                                        addr::Address, 
                                        call::Function, 
                                        len::Int, 
                                        args...)
    visit!(ctx, addr => 1)
    ret, cl, w = propose(call, args...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_cl = Vector{typeof(cl)}(undef, len)
    v_ret[1] = ret
    v_cl[1] = cl
    increment!(ctx, w)
    for i in 2:len
        visit!(ctx, addr => i)
        ret, cl, w = propose(call, v_ret[i-1]...)
        v_ret[i] = ret
        v_cl[i] = cl
        increment!(ctx, w)
    end
    sc = sum(map(v_cl) do cl
                 get_score(cl)
                end)
    add_call!(ctx, addr, VectorizedSite{typeof(markov)}(VectorizedTrace(v_cl), sc, call, args, v_ret))
    return v_ret
end

@inline function (ctx::ProposeContext)(c::typeof(plate), 
                                        addr::Address, 
                                        call::Function, 
                                        args::Vector)
    visit!(ctx, addr => 1)
    len = length(args)
    ret, cl, w = propose(call, args[1]...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_cl = Vector{typeof(cl)}(undef, len)
    v_ret[1] = ret
    v_cl[1] = cl
    increment!(ctx, w)
    for i in 2:len
        visit!(ctx, addr => i)
        ret, cl, w = propose(call, args[i]...)
        v_ret[i] = ret
        v_cl[i] = cl
        increment!(ctx, w)
    end
    sc = sum(map(v_cl) do cl
                 get_score(cl)
                end)
    add_call!(ctx, addr, VectorizedSite{typeof(plate)}(VectorizedTrace(v_cl), sc, call, args, v_ret))
    return v_ret
end

# ------------ Convenience ------------ #

function propose(fn::Function, args...; params = LearnableParameters())
    ctx = Propose(params)
    ret = ctx(fn, args...)
    return ret, GenericCallSite(ctx.tr, ctx.score, fn, args, ret), ctx.weight
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
The convenience `propose` function provides an easy API to the `Propose` context. You can use this function on any of the matching signatures above - it will return the return value `ret`, a `RecordSite` instance specialized to the call, and the score/weight `w`.
""", propose)
