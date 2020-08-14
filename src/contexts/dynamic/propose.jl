# ------------ Choice sites ------------ #

@inline function (ctx::ProposeContext)(call::typeof(rand), 
                                       addr::T, 
                                       d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)
    s = rand(d)
    score = logpdf(d, s)
    add_choice!(ctx, addr, score, s)
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

@inline function (ctx::ProposeContext)(c::typeof(rand),
                                       addr::T,
                                       call::Function,
                                       args...) where T <: Address
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ret, cl, w = propose(ps, call, args...)
    add_call!(ctx, addr, cl)
    return ret
end

# ------------ Convenience ------------ #

function propose(fn::Function, args...)
    ctx = Propose(DynamicTrace())
    ret = ctx(fn, args...)
    return ret, DynamicCallSite(ctx.tr, ctx.score, fn, args, ret), ctx.score
end

function propose(params, fn::Function, args...)
    ctx = Propose(DynamicTrace(), params)
    ret = ctx(fn, args...)
    return ret, DynamicCallSite(ctx.tr, ctx.score, fn, args, ret), ctx.score
end

function propose(fn::typeof(rand), d::Distribution{K}) where K
    ctx = Propose()
    addr = gensym()
    ret = ctx(fn, addr, d)
    return ret, get_sub(ctx.tr, addr), ctx.score
end
