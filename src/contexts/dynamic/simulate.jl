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

@inline function (ctx::SimulateContext)(fn::typeof(learnable), addr::T) where T <: Address
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

# ------------ Convenience ------------ #

function simulate(fn::Function, args...)
    ctx = Simulate(Trace(), Empty())
    ret = ctx(fn, args...)
    return ret, DynamicCallSite(ctx.tr, ctx.score, fn, args, ret)
end

function simulate(params::P, fn::Function, args...) where P <: AddressMap
    ctx = Simulate(Trace(), params)
    ret = ctx(fn, args...)
    return ret, DynamicCallSite(ctx.tr, ctx.score, fn, args, ret)
end

function simulate(fn::typeof(trace), d::Distribution{T}) where T
    ctx = Simulate(Trace(), Empty())
    addr = gensym()
    ret = ctx(trace, addr, d)
    return ret, get_sub(ctx.tr, addr)
end

function simulate(params::P, fn::typeof(trace), d::Distribution{T}) where {P <: AddressMap, T}
    ctx = Simulate(Trace(), params)
    addr = gensym()
    ret = ctx(trace, addr, d)
    return ret, get_sub(ctx.tr, addr)
end

