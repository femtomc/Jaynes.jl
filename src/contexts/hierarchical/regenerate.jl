# ------------ Choice sites ------------ #

@inline function (ctx::RegenerateContext)(call::typeof(rand), 
                                          addr::T, 
                                          d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)
    in_prev_chm = has_choice(ctx.prev, addr)
    in_sel = has_query(ctx.select, addr)
    if in_prev_chm
        prev = get_choice(ctx.prev.trace, addr)
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

@inline function (ctx::RegenerateContext)(fn::typeof(learnable), addr::Address, p::T) where T
    visit!(ctx, addr)
    ret = p
    if has_param(ctx.params, addr)
        ret = get_param(ctx.params, addr)
    end
    return ret
end

# ------------ Black box call sites ------------ #

@inline function (ctx::RegenerateContext)(c::typeof(rand),
                                          addr::T,
                                          call::Function,
                                          args...) where T <: Address
    visit!(ctx, addr)
    ss = get_subselection(ctx, addr)
    prev_call = get_prev(ctx, addr)
    ret, cl, w, retdiff, d = regenerate(ss, prev_call, args...)
    set_call!(ctx.tr, addr, cl)
    increment!(ctx, w)
    return ret
end

