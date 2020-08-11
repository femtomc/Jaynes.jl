# ------------ Choice sites ------------ #

@inline function (ctx::RegenerateContext)(call::typeof(rand), 
                                          addr::T, 
                                          d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)
    in_prev_chm = has_top(ctx.prev, addr)
    in_sel = has_top(ctx.select, addr)
    if in_prev_chm
        prev = get_top(ctx.prev.trace, addr)
        if in_sel
            ret = rand(d)
            add_choice!(ctx.discard, addr, prev)
        else
            ret = prev.val
        end
    end
    score = logpdf(d, ret)
    if in_prev_chm && in_sel
        increment!(ctx, score - prev.score)
    end
    add_choice!(ctx, addr, ChoiceSite(score, ret))
    return ret
end

# ------------ Learnable ------------ #

@inline function (ctx::RegenerateContext)(fn::typeof(learnable), addr::Address)
    visit!(ctx, addr)
    has_top(ctx.params, addr) && return get_top(ctx.params, addr)
    error("(learnable): parameter not provided at address $addr.")
end

# ------------ Fillable ------------ #

@inline function (ctx::RegenerateContext)(fn::typeof(fillable), addr::Address)
    has_top(ctx.select, addr) && return get_top(ctx.select, addr)
    error("(fillable): parameter not provided at address $addr.")
end

# ------------ Black box call sites ------------ #

@inline function (ctx::RegenerateContext)(c::typeof(rand),
                                          addr::T,
                                          call::Function,
                                          args...) where T <: Address
    visit!(ctx, addr)
    ps = get_subparameters(ctx, addr)
    ss = get_subselection(ctx, addr)
    if has_sub(ctx.prev, addr)
        prev_call = get_prev(ctx, addr)
        ret, cl, w, retdiff, d = regenerate(ss, ps, prev_call, args...)
    else
        ret, cl, w = generate(ss, ps, call, args...)
    end
    add_call!(ctx.tr, addr, cl)
    increment!(ctx, w)
    return ret
end
