# ------------ Choice sites ------------ #

@inline function (ctx::GenerateContext)(call::typeof(trace), 
                                        addr::T, 
                                        d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)
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
    haskey(ctx.params, addr) && return getindex(ctx.params, addr)
    error("(learnable): parameter not provided at address $addr.")
end

# ------------ Fillable ------------ #

@inline function (ctx::GenerateContext)(fn::typeof(fillable), addr::Address)
    haskey(ctx.target, addr) && return getindex(ctx.target, addr)
    error("(fillable): parameter not provided at address $addr.")
end

# ------------ Black box call sites ------------ #

@inline function (ctx::GenerateContext)(c::typeof(trace),
                                        addr::T,
                                        call::Function,
                                        args...) where T <: Address
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ss = get_sub(ctx.target, addr)
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

function generate(target::L, fn::typeof(trace), d::Distribution{K}) where {L <: AddressMap, K}
    ctx = Generate(target)
    addr = gensym()
    ret = ctx(fn, addr, d)
    return ret, get_sub(ctx.tr, addr), ctx.weight
end
