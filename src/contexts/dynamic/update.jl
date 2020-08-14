# ------------ Choice sites ------------ #

@inline function (ctx::UpdateContext)(call::typeof(rand), 
                                      addr::T, 
                                      d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)

    # Check if in previous trace's choice map.
    in_prev_chm = has_value(ctx.prev, addr)
    in_prev_chm && begin
        prev = get_sub(ctx.prev, addr)
        prev_ret = get_value(prev)
        prev_score = get_score(prev)
    end

    # Check if in schema.
    in_schema = has_value(ctx.schema, addr)

    # Ret.
    if in_schema
        ret = getindex(ctx.schema, addr)
        in_prev_chm && begin
            set_sub!(ctx.discard, addr, prev)
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
    elseif in_schema
        increment!(ctx, score)
    end
    add_choice!(ctx, addr, score, ret)

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
    has_top(ctx.schema, addr) && return get_top(ctx.schema, addr)
    error("(fillable): parameter not provided at address $addr.")
end

# ------------ Black box call sites ------------ #

@inline function (ctx::UpdateContext)(c::typeof(rand),
                                      addr::T,
                                      call::Function,
                                      args...) where {T <: Address, D <: Diff}
    visit!(ctx, addr)
    ps = get_subparameters(ctx, addr)
    ss = get_subschema(ctx, addr)
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

# ------------ Utilities ------------ #

function update_projection_walk(tr::DynamicTrace,
                                visited::Visitor)
    weight = 0.0
    for (k, v) in shallow_iterator(tr)
        if !(k in visited)
            weight += projection(v, SelectAll())
        end
    end
    weight
end

function update_discard_walk!(d::DynamicDiscard,
                              visited::Visitor,
                              prev::DynamicTrace)
    for (k, v) in shallow_iterator(prev)
        if !(k in visited)
            ss = get_sub(visited, k)
            if isempty(ss)
                set_sub!(d, k, v)
            else
                sd = get_sub(d, k)
                sd = isempty(sd) ? DynamicMap{Value}() : sd
                discard_walk!(sd, ss, v)
                set_sub!(d, k, sd)
            end
        end
    end
end

# ------------ Convenience ------------ #

function update(ctx::UpdateContext, cs::DynamicCallSite, args...) where D <: Diff
    ret = ctx(cs.fn, args...)
    adj_w = update_projection_walk(ctx.tr, ctx.visited)
    update_discard_walk!(ctx.discard, ctx.visited, ctx.tr)
    return ret, DynamicCallSite(ctx.tr, ctx.score - adj_w, cs.fn, args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function update(sel::L, cs::DynamicCallSite) where L <: AddressMap
    argdiffs = NoChange()
    ctx = Update(sel, cs, argdiffs)
    return update(ctx, cs, cs.args...)
end

function update(sel::L, ps::P, cs::DynamicCallSite) where {L <: AddressMap, P <: AddressMap}
    argdiffs = NoChange()
    ctx = UpdateContext(cs, sel, ps, argdiffs)
    return update(ctx, cs, cs.args...)
end

function update(cs::DynamicCallSite, argdiffs::D, new_args...) where D <: Diff
    sel = selection()
    ctx = UpdateContext(cs, sel, argdiffs)
    return update(ctx, cs, new_args...)
end

function update(ps::P, cs::DynamicCallSite, argdiffs::D, new_args...) where {P <: AddressMap, D <: Diff}
    sel = selection()
    ctx = UpdateContext(cs, sel, ps, argdiffs)
    return update(ctx, cs, new_args...)
end

function update(sel::L, cs::DynamicCallSite, argdiffs::D, new_args...) where {L <: AddressMap, D <: Diff}
    ctx = UpdateContext(cs, sel, argdiffs)
    return update(ctx, cs, new_args...)
end

function update(sel::L, ps::P, cs::DynamicCallSite, argdiffs::D, new_args...) where {L <: AddressMap, P <: AddressMap, D <: Diff}
    ctx = UpdateContext(cs, sel, ps, argdiffs)
    return update(ctx, cs, new_args...)
end
