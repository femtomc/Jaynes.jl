# ------------ Choice sites ------------ #

@inline function (ctx::GenerateContext)(call::typeof(rand), 
                                        addr::T, 
                                        d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)
    println(addr)
    if has_value(ctx.target, addr)
        s = getindex(ctx.target, addr)
        score = logpdf(d, s)
        add_choice!(ctx, addr, score, s)
        increment!(ctx, score)
    else
        s = rand(d)
        add_choice!(ctx, addr, logpdf(d, s), s)
    end
    return s
end

# ------------ Learnable ------------ #

@inline function (ctx::GenerateContext)(fn::typeof(learnable), addr::Address)
    visit!(ctx, addr)
    has_top(ctx.params, addr) && return get_top(ctx.params, addr)
    error("(learnable): parameter not provided at address $addr.")
end

# ------------ Fillable ------------ #

@inline function (ctx::GenerateContext)(fn::typeof(fillable), addr::Address)
    has_top(ctx.target, addr) && return get_top(ctx.target, addr)
    error("(fillable): parameter not provided at address $addr.")
end

# ------------ Black box call sites ------------ #

@inline function (ctx::GenerateContext)(c::typeof(rand),
                                        addr::T,
                                        call::Function,
                                        args...) where T <: Address
    visit!(ctx, addr)
    ps = get_subparameters(ctx, addr)
    ss = get_subtarget(ctx, addr)
    ret, cl, w = generate(ss, ps, call, args...)
    add_call!(ctx, addr, cl)
    increment!(ctx, w)
    return ret
end

# ------------ Convenience ------------ #

function generate(target::L, fn::Function, args...) where L <: AddressMap
    ctx = Generate(target)
    ret = ctx(fn, args...)
    return ret, DynamicCallSite(ctx.tr, ctx.score, fn, args, ret), ctx.weight
end

function generate(target::L, params, fn::Function, args...) where L <: AddressMap
    ctx = Generate(target, params)
    ret = ctx(fn, args...)
    return ret, DynamicCallSite(ctx.tr, ctx.score, fn, args, ret), ctx.weight
end

function generate(target::L, fn::typeof(rand), d::Distribution{K}) where {L <: AddressMap, K}
    ctx = Generate(target)
    addr = gensym()
    ret = ctx(fn, addr, d)
    return ret, get_top(ctx.tr, addr), ctx.weight
end
