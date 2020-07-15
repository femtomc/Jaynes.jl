mutable struct GenerateContext{T <: Trace, K <: ConstrainedSelection} <: ExecutionContext
    tr::T
    select::K
    weight::Float64
    score::Float64
    visited::Visitor
    params::LearnableParameters
    GenerateContext(tr::T, select::K) where {T <: Trace, K <: ConstrainedSelection} = new{T, K}(tr, select, 0.0, 0.0, Visitor(), LearnableParameters())
    GenerateContext(tr::T, select::K, params) where {T <: Trace, K <: ConstrainedSelection} = new{T, K}(tr, select, 0.0, 0.0, Visitor(), params)
end
Generate(select::ConstrainedSelection) = GenerateContext(Trace(), select)
Generate(select::ConstrainedSelection, params) = GenerateContext(Trace(), select, params)
Generate(tr::Trace, select::ConstrainedSelection) = GenerateContext(tr, select)

# ------------ Choice sites ------------ #

@inline function (ctx::GenerateContext)(call::typeof(rand), 
                                        addr::T, 
                                        d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)
    if has_query(ctx.select, addr)
        s = get_query(ctx.select, addr)
        score = logpdf(d, s)
        add_choice!(ctx, addr, ChoiceSite(score, s))
        increment!(ctx, score)
    else
        s = rand(d)
        add_choice!(ctx, addr, ChoiceSite(logpdf(d, s), s))
    end
    return s
end

@inline function (ctx::GenerateContext)(c::typeof(plate), 
                                        addr::T, 
                                        d::Distribution{K},
                                        len::Int) where {T <: Address, K}
    v_ret = Vector{eltype(d)}(undef, len)
    v_cs = Vector{ChoiceSite{eltype(d)}}(undef, len)
    ss = get_subselection(ctx, addr)
    for i in 1:len
        visit!(ctx, addr => i)
        if has_query(ss, i)
            s = get_query(ss, i)
            score = logpdf(d, s)
            cs = ChoiceSite(score, s)
            increment!(ctx, score)
        else
            s = rand(d)
            score = logpdf(d, s)
            cs = ChoiceSite(score, s)
        end
        v_ret[i] = s
        v_cs[i] = cs
    end
    sc = sum(map(v_cs) do cs
                 get_score(cs)
             end)
    add_call!(ctx, addr, VectorizedSite{typeof(markov)}(VectorizedTrace(v_cs), sc, d, (), v_ret))
    return v_ret
end

# ------------ Learnable ------------ #

@inline function (ctx::GenerateContext)(fn::typeof(learnable), addr::Address, p::T) where T
    visit!(ctx, addr)
    ret = p
    if has_param(ctx.params, addr)
        ret = get_param(ctx.params, addr)
    end
    ctx.tr.params[addr] = ParameterSite(ret)
    return ret
end

# ------------ Black box call sites ------------ #

@inline function (ctx::GenerateContext)(c::typeof(rand),
                                        addr::T,
                                        call::Function,
                                        args...) where T <: Address
    visit!(ctx, addr)
    ss = get_subselection(ctx, addr)
    ret, cl, w = generate(ss, call, args...)
    add_call!(ctx, addr, cl)
    increment!(ctx, w)
    return ret
end

# ------------ Vectorized call sites ------------ #

@inline function (ctx::GenerateContext)(c::typeof(markov), 
                                        addr::Address, 
                                        call::Function, 
                                        len::Int, 
                                        args...)
    visit!(ctx, addr => 1)
    ss = get_subselection(ctx, addr => 1)
    ret, cl, w = generate(ss, call, args...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_cl = Vector{typeof(cl)}(undef, len)
    v_ret[1] = ret
    v_cl[1] = cl
    increment!(ctx, w)
    for i in 2:len
        visit!(ctx, addr => i)
        ss = get_subselection(ctx, addr => i)
        ret, cl, w = generate(ss, call, v_ret[i-1]...)
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

@inline function (ctx::GenerateContext)(c::typeof(plate), 
                                        addr::Address, 
                                        call::Function, 
                                        args::Vector)
    visit!(ctx, addr => 1)
    len = length(args)
    ss = get_subselection(ctx, addr => 1)
    ret, cl, w = generate(ss, call, args[1]...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_cl = Vector{typeof(cl)}(undef, len)
    v_ret[1] = ret
    v_cl[1] = cl
    increment!(ctx, w)
    for i in 2:len
        visit!(ctx, addr => i)
        ss = get_subselection(ctx, addr => i)
        ret, cl, w = generate(ss, call, args[i]...)
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

# ------------ Convenience ------------ #

function generate(sel::L, fn::Function, args...; params = LearnableParameters()) where L <: ConstrainedSelection
    ctx = Generate(sel, params)
    ret = ctx(fn, args...)
    return ret, GenericCallSite(ctx.tr, ctx.score, fn, args, ret), ctx.weight
end

function generate(sel::L, fn::typeof(rand), d::Distribution{K}) where {L <: ConstrainedSelection, K}
    ctx = Generate(sel)
    addr = gensym()
    ret = ctx(fn, addr, d)
    return ret, get_choice(ctx.tr, addr), ctx.weight
end

function generate(sel::L, fn::typeof(markov), call::Function, len::Int, args...) where L <: ConstrainedSelection
    addr = gensym()
    v_sel = selection(addr => sel)
    ctx = Generate(v_sel)
    ret = ctx(fn, addr, call, len, args...)
    return ret, get_call(ctx.tr, addr), ctx.weight
end

function generate(sel::L, fn::typeof(plate), call::Function, args::Vector) where L <: ConstrainedSelection
    ctx = Generate(sel)
    addr = gensym()
    ret = ctx(fn, addr, call, args)
    return ret, get_call(ctx.tr, addr), ctx.weight
end

function generate(sel::L, fn::typeof(plate), d::Distribution{K}, len::Int) where {L <: ConstrainedSelection, K}
    addr = gensym()
    v_sel = selection(addr => sel)
    ctx = Generate(v_sel)
    ret = ctx(fn, addr, d, len)
    return ret, get_call(ctx.tr, addr), ctx.weight
end

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct GenerateContext{T <: Trace, K <: ConstrainedSelection} <: ExecutionContext
     tr::T
     select::K
     weight::Float64
     score::Float64
     visited::Visitor
     params::LearnableParameters
end
```
`GenerateContext` is used to generate traces, as well as record and accumulate likelihood weights given observations at addressed randomness.

Inner constructors:
```julia
GenerateContext(tr::T, select::K) where {T <: Trace, K <: ConstrainedSelection} = new{T, K}(tr, select, 0.0, Visitor(), LearnableParameters())
GenerateContext(tr::T, select::K, params) where {T <: Trace, K <: ConstrainedSelection} = new{T, K}(tr, select, 0.0, Visitor(), params)
```
Outer constructors:
```julia
Generate(select::ConstrainedSelection) = GenerateContext(Trace(), select)
Generate(select::ConstrainedSelection, params) = GenerateContext(Trace(), select, params)
Generate(tr::Trace, select::ConstrainedSelection) = GenerateContext(tr, select)
```
""", GenerateContext)

@doc(
"""
```julia
ret, cl, w = generate(sel::L, fn::Function, args...; params = LearnableParameters()) where L <: ConstrainedSelection
ret, cs, w = generate(sel::L, fn::typeof(rand), d::Distribution{K}) where {L <: ConstrainedSelection, K}
ret, v_cl, w = generate(sel::L, fn::typeof(markov), call::Function, len::Int, args...) where L <: ConstrainedSelection
ret, v_cl, w = generate(sel::L, fn::typeof(plate), call::Function, args::Vector) where L <: ConstrainedSelection
ret, v_cl, w = generate(sel::L, fn::typeof(plate), d::Distribution{K}, len::Int) where {L <: ConstrainedSelection, K}
```
The convenience `generate` function provides an easy API to the `Generate` context. You can use this function on any of the matching signatures above - it will return the return value `ret`, a `RecordSite` instance specialized to the call, and the score/weight `w` computed with respect to the constraints `sel`.
""", generate)

