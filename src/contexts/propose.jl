mutable struct ProposeContext{T <: Trace} <: ExecutionContext
    tr::T
    weight::Float64
    visited::Visitor
    params::LearnableParameters
    ProposeContext(tr::T) where T <: Trace = new{T}(tr, 0.0, Visitor(), LearnableParameters())
end
Propose() = ProposeContext(Trace())

# ------------ Choice sites ------------ #

@inline function (ctx::ProposeContext)(call::typeof(rand), 
                                       addr::T, 
                                       d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)
    s = rand(d)
    score = logpdf(d, s)
    add_choice!(ctx.tr, addr, ChoiceSite(score, s))
    increment!(ctx, score)
    return s
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
    add_call!(ctx.tr, addr, cl)
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
    v_tr = Vector{HierarchicalTrace}(undef, len)
    v_ret[1] = ret
    v_tr[1] = cl.trace
    increment!(ctx, w)
    for i in 2:len
        visit!(ctx, addr => i)
        ret, cl, w = propose(call, v_ret[i-1]...)
        v_ret[i] = ret
        v_tr[i] = cl.trace
        increment!(ctx, w)
    end
    sc = sum(map(v_tr) do tr
                    score(tr)
                end)
    add_call!(ctx.tr, addr, VectorizedCallSite{typeof(markov)}(v_tr, sc, call, args, v_ret))
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
    v_tr = Vector{HierarchicalTrace}(undef, len)
    v_ret[1] = ret
    v_tr[1] = cl.trace
    for i in 2:len
        ret, cl, w = propose(call, args[i]...)
        v_ret[i] = ret
        v_tr[i] = p_ctx.tr
    end
    sc = sum(map(v_tr) do tr
                    score(tr)
                end)
    add_call!(ctx.tr, addr, VectorizedCallSite{typeof(plate)}(v_tr, sc, call, args, v_ret))
    return v_ret
end

# ------------ Convenience ------------ #

function propose(fn::Function, args...)
    ctx = Propose()
    ret = ctx(fn, args...)
    return ret, BlackBoxCallSite(ctx.tr, fn, args, ret), ctx.weight
end

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct ProposeContext{T <: Trace} <: ExecutionContext
    tr::T
    weight::Float64
    params::LearnableParameters
end
```

`ProposeContext` is used to generate traces for inference algorithms which use custom proposals. `ProposeContext` instances can be passed sets of `LearnableParameters` to configure the propose with parameters which have been learned by differentiable programming.

Inner constructors:

```julia
ProposeContext(tr::T) where T <: Trace = new{T}(tr, 0.0, LearnableParameters())
```

Outer constructors:

```julia
Propose() = ProposeContext(Trace())
```
""", ProposeContext)

