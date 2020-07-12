mutable struct SimulateContext{T <: Trace} <: ExecutionContext
    tr::T
    visited::Visitor
    params::LearnableParameters
    SimulateContext(params) = new{HierarchicalTrace}(Trace(), Visitor(), params)
end

# ------------ Choice sites ------------ #

@inline function (ctx::SimulateContext)(call::typeof(rand), 
                                        addr::T, 
                                        d::Distribution{K}) where {T <: Address, K}
    s = rand(d)
    add_choice!(ctx.tr, addr, ChoiceSite(logpdf(d, s), s))
    visit!(ctx.visited, addr)
    return s
end

# ------------ Learnable ------------ #

@inline function (ctx::SimulateContext)(fn::typeof(learnable), addr::Address, p::T) where T
    visit!(ctx.visited, addr)
    ret = p
    if has_param(ctx.params, addr)
        ret = get_param(ctx.params, addr)
    end
    ctx.tr.params[addr] = ParameterSite(ret)
    return ret
end

# ------------ Call sites ------------ #

@inline function (ctx::SimulateContext)(c::typeof(rand),
                                        addr::T,
                                        call::Function,
                                        args...) where T <: Address
    ss = get_subselection(ctx, addr)
    ret, cl = simulate(ss, call, args...)
    add_call!(ctx.tr, addr, cl)
    return ret
end

@inline function (ctx::SimulateContext)(c::typeof(foldr), 
                                        fn::typeof(rand), 
                                        addr::Address, 
                                        call::Function, 
                                        len::Int, 
                                        args...)
    ug_ctx = Simulate(Trace(), get_sub(ctx.select, addr => 1))
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

@inline function (ctx::SimulateContext)(c::typeof(map), 
                                        fn::typeof(rand), 
                                        addr::Address, 
                                        call::Function, 
                                        args::Vector)
    ug_ctx = Simulate(Trace(), get_sub(ctx.select, addr => 1))
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
function simulate(fn::Function, args...; params = LearnableParameters())
    ctx = SimulateContext(params)
    ret = ctx(fn, args...)
    return ret, BlackBoxCallSite(ctx.tr, fn, args, ret)
end

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
