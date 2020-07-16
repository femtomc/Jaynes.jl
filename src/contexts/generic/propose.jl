# ------------ Choice sites ------------ #

@inline function (ctx::ProposeContext)(call::typeof(rand), 
                                       addr::T, 
                                       d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)
    s = rand(d)
    score = logpdf(d, s)
    add_choice!(ctx, addr, ChoiceSite(score, s))
    increment!(ctx, score)
    return s
end

# ------------ Learnable ------------ #

@inline function (ctx::ProposeContext)(fn::typeof(learnable), addr::Address, p::T) where T
    visit!(ctx, addr)
    ret = p
    if has_param(ctx.params, addr)
        ret = get_param(ctx.params, addr)
    end
    return ret
end

# ------------ Black box call sites ------------ #

@inline function (ctx::ProposeContext)(c::typeof(rand),
                                        addr::T,
                                        call::Function,
                                        args...) where T <: Address
    visit!(ctx, addr)
    ret, cl, w = propose(call, args...)
    add_call!(ctx, addr, cl)
    increment!(ctx, w)
    return ret
end
