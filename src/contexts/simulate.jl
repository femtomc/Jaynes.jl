mutable struct SimulateContext{T <: Trace} <: ExecutionContext
    tr::T
    score::Float64
    visited::Visitor
    params::LearnableParameters
    SimulateContext(params) = new{HierarchicalTrace}(Trace(), 0.0, Visitor(), params)
end

# ------------ Choice sites ------------ #

@inline function (ctx::SimulateContext)(call::typeof(rand), 
                                        addr::T, 
                                        d::Distribution{K}) where {T <: Address, K}
    visit!(ctx.visited, addr)
    s = rand(d)
    add_choice!(ctx, addr, ChoiceSite(logpdf(d, s), s))
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
    add_call!(ctx, addr, cl)
    return ret
end

# ------------ Vectorized call sites ------------ #

@inline function (ctx::SimulateContext)(c::typeof(markov), 
                                        addr::Address, 
                                        call::Function, 
                                        len::Int, 
                                        args...)
    visit!(ctx, addr => 1)
    ret, cl = simulate(call, args...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_cl = Vector{typeof(cl)}(undef, len)
    v_ret[1] = ret
    v_cl[1] = cl
    for i in 2:len
        visit!(ctx, addr => i)
        ret, cl = simulate(call, v_ret[i-1]...)
        v_ret[i] = ret
        v_cl[i] = cl
    end
    sc = sum(map(v_cl) do cl
                 get_score(cl)
             end)
    add_call!(ctx, addr, VectorizedSite{typeof(markov)}(VectorizedTrace(v_cl), sc, call, args, v_ret))
    return v_ret
end

@inline function (ctx::SimulateContext)(c::typeof(plate), 
                                        addr::Address, 
                                        call::Function, 
                                        args::Vector)
    visit!(ctx, addr => 1)
    len = length(args)
    ret, cl = simulate(call, args[1]...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_cl = Vector{typeof(cl)}(undef, len)
    v_ret[1] = ret
    v_cl[1] = cl
    for i in 2:len
        visit!(ctx, addr => i)
        ret, cl = simulate(call, args[i]...)
        v_ret[i] = ret
        v_cl[i] = cl
    end
    sc = sum(map(v_cl) do cl
                 get_score(cl)
             end)
    add_call!(ctx, addr, VectorizedSite{typeof(markov)}(VectorizedTrace(v_cl), sc, call, args, v_ret))
    return v_ret
end

# ------------ Convenience ------------ #

function simulate(fn::Function, args...; params = LearnableParameters())
    ctx = SimulateContext(params)
    ret = ctx(fn, args...)
    return ret, BlackBoxCallSite(ctx.tr, ctx.score, fn, args, ret)
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
