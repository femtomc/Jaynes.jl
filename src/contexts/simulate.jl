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

# ------------ Black box call sites ------------ #

@inline function (ctx::SimulateContext)(c::typeof(rand),
                                        addr::T,
                                        call::Function,
                                        args...) where T <: Address
    ss = get_subselection(ctx, addr)
    ret, cl = simulate(ss, call, args...)
    add_call!(ctx.tr, addr, cl)
    return ret
end

# ------------ Vectorized call sites ------------ #

@inline function (ctx::SimulateContext)(c::typeof(markov), 
                                        addr::Address, 
                                        call::Function, 
                                        len::Int, 
                                        args...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_tr = Vector{HierarchicalTrace}(undef, len)
    ret = args
    for i in 1:len
        visit!(ctx, addr => i)
        ss = get_subselection(ctx, addr => i)
        ret, cl = simulate(ss, call, ret...)
        v_ret[i] = ret
        v_tr[i] = cl.tr
    end
    sc = sum(map(v_tr) do tr
                 get_score(tr)
             end)
    add_call!(ctx.tr, addr, VectorizedCallSite{typeof(markov)}(v_tr, sc, call, args, v_ret))
    return v_ret
end

@inline function (ctx::SimulateContext)(c::typeof(plate), 
                                        addr::Address, 
                                        call::Function, 
                                        args::Vector)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_tr = Vector{HierarchicalTrace}(undef, len)
    for i in 1:len
        visit!(ctx, addr => i)
        ss = get_subselection(ctx, addr => i)
        ret, cl = simulate(ss, call, args[i]...)
        v_ret[i] = ret
        v_tr[i] = cl.tr
    end
    sc = sum(map(v_tr) do tr
                 get_score(tr)
             end)
    add_call!(ctx.tr, addr, VectorizedCallSite{typeof(markov)}(v_tr, sc, call, args, v_ret))
    return v_ret
end

# ------------ Convenience ------------ #

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
