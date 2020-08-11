# ------------ Choice sites ------------ #

@inline function (ctx::UpdateContext)(call::typeof(rand), 
                                      addr::T, 
                                      d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)

    # Check if in previous trace's choice map.
    in_prev_chm = has_top(ctx.prev.trace, addr)
    in_prev_chm && begin
        prev = get_top(ctx.prev.trace, addr)
        prev_ret = prev.val
        prev_score = prev.score
    end

    # Check if in selection.
    in_selection = has_top(ctx.select, addr)

    # Ret.
    if in_selection
        ret = get_top(ctx.select, addr)
        in_prev_chm && begin
            add_choice!(ctx.discard, addr, prev)
        end
    elseif in_prev_chm
        ret = prev_ret
    else
        ret = rand(d)
    end

    # Update.
    score = logpdf(d, ret)
    if in_prev_chm
        increment!(ctx, score - prev_score)
    elseif in_selection
        increment!(ctx, score)
    end
    add_choice!(ctx, addr, ChoiceSite(score, ret))

    return ret
end

# ------------ Learnable ------------ #

@inline function (ctx::UpdateContext)(fn::typeof(learnable), addr::Address)
    visit!(ctx, addr)
    has_top(ctx.params, addr) && return get_top(ctx.params, addr)
    error("(learnable): parameter not provided at address $addr.")
end

# ------------ Fillable ------------ #

@inline function (ctx::UpdateContext)(fn::typeof(fillable), addr::Address)
    has_top(ctx.select, addr) && return get_top(ctx.select, addr)
    error("(fillable): parameter not provided at address $addr.")
end

# ------------ Black box call sites ------------ #

@inline function (ctx::UpdateContext)(c::typeof(rand),
                                      addr::T,
                                      call::Function,
                                      args...) where {T <: Address, D <: Diff}
    visit!(ctx, addr)
    ps = get_subparameters(ctx, addr)
    ss = get_subselection(ctx, addr)
    if has_sub(ctx.prev, addr)
        prev = get_prev(ctx, addr)
        ret, cl, w, rd, d = update(ss, ps, prev, UndefinedChange(), args...)
    else
        ret, cl, w = generate(ss, ps, call, args...)
    end
    add_call!(ctx, addr, cl)
    increment!(ctx, w)
    return ret
end
