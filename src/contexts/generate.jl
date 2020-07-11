mutable struct GenerateContext{T <: Trace, K <: ConstrainedSelection} <: ExecutionContext
    tr::T
    select::K
    weight::Float64
    visited::Visitor
    params::LearnableParameters
    GenerateContext(tr::T, select::K) where {T <: Trace, K <: ConstrainedSelection} = new{T, K}(tr, select, 0.0, Visitor(), LearnableParameters())
    GenerateContext(tr::T, select::K, params) where {T <: Trace, K <: ConstrainedSelection} = new{T, K}(tr, select, 0.0, Visitor(), params)
end
Generate(select::ConstrainedSelection) = GenerateContext(Trace(), select)
Generate(select::ConstrainedSelection, params) = GenerateContext(Trace(), select, params)
Generate(tr::Trace, select::ConstrainedSelection) = GenerateContext(tr, select)

# ------------ Choice sites ------------ #

@inline function (ctx::GenerateContext)(call::typeof(rand), 
                                        addr::T, 
                                        d::Distribution{K}) where {T <: Address, K}
    if has_query(ctx.select, addr)
        s = get_query(ctx.select, addr)
        score = logpdf(d, s)
        add_choice!(ctx.tr, addr, ChoiceSite(score, s))
        increment!(ctx, score)
    else
        s = rand(d)
        add_choice!(ctx.tr, addr, ChoiceSite(logpdf(d, s), s))
    end
    visit!(ctx.visited, addr)
    return s
end

# ------------ Learnable ------------ #

@inline function (ctx::GenerateContext)(fn::typeof(learnable), addr::Address, p::T) where T
    visit!(ctx.visited, addr)
    ret = p
    if has_param(ctx.params, addr)
        ret = get_param(ctx.params, addr)
    end
    ctx.tr.params[addr] = ParameterSite(ret)
    return ret
end

# ------------ Call sites ------------ #

@inline function (ctx::GenerateContext)(c::typeof(rand),
                                        addr::T,
                                        call::Function,
                                        args...) where T <: Address
    ss = get_subselection(ctx, addr)
    ret, cl, w = generate(ss, call, args...)
    add_call!(ctx.tr, addr, cl)
    increment!(ctx, w)
    return ret
end

@inline function (ctx::GenerateContext)(c::typeof(foldr), 
                                        fn::typeof(rand), 
                                        addr::Address, 
                                        call::Function, 
                                        len::Int, 
                                        args...)
    ug_ctx = Generate(Trace(), get_sub(ctx.select, addr => 1))
    ret = ug_ctx(call, args...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_tr = Vector{HierarchicalTrace}(undef, len)
    v_ret[1] = ret
    v_tr[1] = ug_ctx.tr
    ctx.weight += ug_ctx.weight
    set_sub!(ctx.visited, addr => 1, ug_ctx.visited)
    for i in 2:len
        ug_ctx.select = get_sub(ctx.select, addr => i)
        ug_ctx.tr = Trace()
        ug_ctx.visited = Visitor()
        ret = ug_ctx(call, v_ret[i-1]...)
        v_ret[i] = ret
        v_tr[i] = ug_ctx.tr
        ctx.weight += ug_ctx.weight
        set_sub!(ctx.visited, addr => i, ug_ctx.visited)
    end
    sc = sum(map(v_tr) do tr
                 get_score(tr)
             end)
    add_call!(ctx.tr, addr, VectorizedCallSite{typeof(foldr)}(v_tr, sc, call, args, v_ret))
    return v_ret
end

@inline function (ctx::GenerateContext)(c::typeof(map), 
                                        fn::typeof(rand), 
                                        addr::Address, 
                                        call::Function, 
                                        args::Vector)
    ug_ctx = Generate(Trace(), get_sub(ctx.select, addr => 1))
    ret = ug_ctx(call, args[1]...)
    len = length(args)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_tr = Vector{HierarchicalTrace}(undef, len)
    v_ret[1] = ret
    v_tr[1] = ug_ctx.tr
    ctx.weight += ug_ctx.weight
    set_sub!(ctx.visited, addr => 1, ug_ctx.visited)
    for i in 2:len
        ug_ctx.select = get_sub(ctx.select, addr => i)
        ug_ctx.tr = Trace()
        ug_ctx.visited = Visitor()
        ret = ug_ctx(call, args[i]...)
        v_ret[i] = ret
        v_tr[i] = ug_ctx.tr
        ctx.weight += ug_ctx.weight
        set_sub!(ctx.visited, addr => i, ug_ctx.visited)
    end
    sc = sum(map(v_tr) do tr
                 get_score(tr)
             end)
    add_call!(ctx.tr, addr, VectorizedCallSite{typeof(map)}(v_tr, sc, call, args, v_ret))
    return v_ret
end

# Convenience.
function generate(sel::L, fn::Function, args...; params = LearnableParameters()) where L <: ConstrainedSelection
    ctx = Generate(sel, params)
    ret = ctx(fn, args...)
    return ret, BlackBoxCallSite(ctx.tr, fn, args, ret), ctx.weight
end

function generate(sel::L, fn::typeof(foldr), r::typeof(rand), addr::Symbol, call::Function, args...) where L <: ConstrainedSelection
    ctx = Generate(sel)
    ret = ctx(fn, r, addr, args...)
    return ret, ctx.tr.chm[addr], ctx.weight
end

function generate(sel::L, fn::typeof(foldr), call::Function, len::Int, args...) where L <: ConstrainedSelection
    anon_sel = ConstrainedHierarchicalSelection()
    addr = gensym()
    push!(anon_sel, addr, sel)
    ctx = Generate(anon_sel)
    ret = ctx(fn, rand, addr, call, len, args...)
    return ret, ctx.tr.chm[addr], ctx.weight
end

function generate(sel::L, fn::typeof(map), r::typeof(rand), addr::Symbol, call::Function, args::Vector) where L <: ConstrainedSelection
    ctx = Generate(sel)
    ret = ctx(fn, r, addr, call, args)
    return ret, ctx.tr.chm[addr], ctx.weight
end

function generate(sel::L, fn::typeof(map), call::Function, args::Vector) where L <: ConstrainedSelection
    anon_sel = ConstrainedHierarchicalSelection()
    addr = gensym()
    push!(anon_sel, addr, sel)
    ctx = Generate(anon_sel)
    ret = ctx(fn, rand, addr, call, args)
    return ret, ctx.tr.chm[addr], ctx.weight
end

function generate(fn, args...)
    return generate(ConstrainedHierarchicalSelection(), fn, args...)
end

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct GenerateContext{T <: Trace, K <: ConstrainedSelection} <: ExecutionContext
    tr::T
    select::K
    weight::Float64
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
Outer constructs:
```julia
Generate(select::ConstrainedSelection) = GenerateContext(Trace(), select)
Generate(select::ConstrainedSelection, params) = GenerateContext(Trace(), select, params)
Generate(tr::Trace, select::ConstrainedSelection) = GenerateContext(tr, select)
```
""", GenerateContext)
