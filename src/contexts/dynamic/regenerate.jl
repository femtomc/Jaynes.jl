# ------------ Choice sites ------------ #

@inline function (ctx::RegenerateContext)(call::typeof(rand), 
                                          addr::T, 
                                          d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)
    in_prev_chm = haskey(get_trace(ctx.prev), addr)
    in_sel = haskey(ctx.select, addr)
    if in_prev_chm
        prev = get_sub(get_trace(ctx.prev), addr)
        if in_sel
            ret = rand(d)
            set_sub!(ctx.discard, addr, prev)
        else
            ret = prev.val
        end
    end
    score = logpdf(d, ret)
    if in_prev_chm && in_sel
        increment!(ctx, score - prev.score)
    end
    add_choice!(ctx, addr, score, ret)
    return ret
end

# ------------ Learnable ------------ #

@inline function (ctx::RegenerateContext)(fn::typeof(learnable), addr::Address)
    visit!(ctx, addr)
    haskey(ctx.params, addr) && return getindex(ctx.params, addr)
    error("(learnable): parameter not provided at address $addr.")
end

# ------------ Fillable ------------ #

@inline function (ctx::RegenerateContext)(fn::typeof(fillable), addr::Address)
    haskey(ctx.select, addr) && return getindex(ctx.select, addr)
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
    if haskey(get_trace(ctx.prev), addr)
        prev_call = get_prev(ctx, addr)
        ret, cl, w, retdiff, d = regenerate(ss, ps, prev_call, args...)
    else
        ret, cl, w = generate(ss, ps, call, args...)
    end
    add_call!(ctx.tr, addr, cl)
    increment!(ctx, w)
    return ret
end

# ------------ Utilities ------------ #

function regenerate_projection_walk(tr::DynamicTrace,
                                visited::Visitor)
    weight = 0.0
    for (k, v) in shallow_iterator(tr)
        if !(k in visited)
            weight += projection(v, SelectAll())
        end
    end
    weight
end

function regenerate_discard_walk!(d::DynamicDiscard,
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

function regenerate(ctx::RegenerateContext, cs::DynamicCallSite, new_args...)
    ret = ctx(cs.fn, new_args...)
    adj_w = regenerate_projection_walk(ctx.tr, ctx.visited)
    regenerate_discard_walk!(ctx.discard, ctx.visited, ctx.tr)
    return ret, DynamicCallSite(ctx.tr, ctx.score - adj_w, cs.fn, new_args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function regenerate(sel::L, cs::DynamicCallSite) where L <: Target
    argdiffs = NoChange()
    ctx = Regenerate(sel, cs, argdiffs)
    return regenerate(ctx, cs, cs.args...)
end

function regenerate(sel::L, ps::P, cs::DynamicCallSite) where {L <: Target, P <: AddressMap}
    argdiffs = NoChange()
    ctx = Regenerate(cs, sel, ps, argdiffs)
    return regenerate(ctx, cs, cs.args...)
end

function regenerate(sel::L, cs::DynamicCallSite, argdiffs::D, new_args...) where {L <: Target, D <: Diff}
    ctx = Regenerate(cs, sel, argdiffs)
    return regenerate(ctx, cs, new_args...)
end

function regenerate(sel::L, ps::P, cs::DynamicCallSite, argdiffs::D, new_args...) where {L <: Target, P <: AddressMap, D <: Diff}
    ctx = Regenerate(cs, sel, ps, argdiffs)
    return regenerate(ctx, cs, new_args...)
end
