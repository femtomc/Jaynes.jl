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

# ------------ Fillable ------------ #

@inline function (ctx::ProposeContext)(fn::typeof(fillable), addr::Address)
    has_sub(ctx.select, addr) && return get_sub(ctx.select, addr)
    error("(fillable): parameter not provided at address $addr.")
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

# ------------ Convenience ------------ #

function propose(fn::Function, args...)
    ctx = Propose(DynamicMap{Value}(), Empty())
    ret = ctx(fn, args...)
    return ret, ctx.map, ctx.score
end

function propose(params, fn::Function, args...)
    ctx = Propose(DynamicMap{Value}(), params)
    ret = ctx(fn, args...)
    return ret, ctx.map, ctx.score
end

function propose(fn::typeof(trace), d::Distribution{K}) where K
    ctx = Propose()
    addr = gensym()
    ret = ctx(fn, addr, d)
    return ret, get_sub(ctx.map, addr), ctx.score
end
