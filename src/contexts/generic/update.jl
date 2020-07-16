# ------------ Choice sites ------------ #

@inline function (ctx::UpdateContext)(call::typeof(rand), 
                                      addr::T, 
                                      d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)

    # Check if in previous trace's choice map.
    in_prev_chm = has_choice(ctx.prev.trace, addr)
    in_prev_chm && begin
        prev = get_choice(ctx.prev.trace, addr)
        prev_ret = prev.val
        prev_score = prev.score
    end

    # Check if in selection.
    in_selection = has_query(ctx.select, addr)

    # Ret.
    if in_selection
        ret = get_query(ctx.select, addr)
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

# ------------ Black box call sites ------------ #

@inline function (ctx::UpdateContext)(c::typeof(rand),
                                      addr::T,
                                      call::Function,
                                      args...) where {T <: Address, D <: Diff}
    visit!(ctx, addr)
    has_addr = has_choice(ctx.prev.trace, addr)
    if has_addr
        cs = get_prev(ctx, addr)
        ss = get_subselection(ctx, addr)

        # TODO: Mjolnir.
        ret, new_site, lw, retdiff, discard = update(ss, cs, args...)

        add_choice!(ctx.discard, addr, CallSite(discard, cs.score, cs.fn, cs.args, cs.ret))
    else
        ss = get_subselection(ctx, addr)
        ret, new_site, lw = generate(ss, call, args...)
    end
    add_call!(ctx, addr, new_site)
    increment!(ctx, lw)
    return ret
end
