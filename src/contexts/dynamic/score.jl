# ------------ Choice sites ------------ #

@inline function (ctx::ScoreContext)(call::typeof(rand), 
                                     addr::T, 
                                     d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)
    has_sub(ctx.select, addr) || error("ScoreError: constrained selection must provide constraints for all possible addresses in trace. Missing at address $addr.")
    val = getindex(ctx.select, addr)
    increment!(ctx, logpdf(d, val))
    return val
end

# ------------ Learnable ------------ #

@inline function (ctx::ScoreContext)(fn::typeof(learnable), addr::Address)
    visit!(ctx, addr)
    has_sub(ctx.params, addr) && return getindex(ctx.params, addr)
    error("Parameter not provided at address $addr.")
end

# ------------ Fillable ------------ #

@inline function (ctx::ScoreContext)(fn::typeof(fillable), addr::Address)
    has_sub(ctx.select, addr) && return getindex(ctx.select, addr)
    error("(fillable): parameter not provided at address $addr.")
end

# ------------ Call sites ------------ #

@inline function (ctx::ScoreContext)(c::typeof(rand),
                                     addr::T,
                                     call::Function,
                                     args...) where T <: Address
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ss = get_sub(ctx.schema, addr)
    ret, w = score(ss, ps, call, args...) 
    increment!(ctx, w)
    return ret
end

# ------------ Convenience ------------ #

function score(sel::L, fn::Function, args...) where L <: AddressMap
    ctx = Score(sel)
    ret = ctx(fn, args...)
    b, missed = compare(sel.query, ctx.visited)
    b || error("ScoreError: did not visit all constraints in selection.\nDid not visit: $(missed).")
    return ret, ctx.weight
end

function score(sel::L, params, fn::Function, args...) where L <: AddressMap
    ctx = Score(sel, params)
    ret = ctx(fn, args...)
    b, missed = compare(sel.query, ctx.visited)
    b || error("ScoreError: did not visit all constraints in selection.\nDid not visit: $(missed).")
    return ret, ctx.weight
end

function score(sel::L, fn::typeof(rand), d::Distribution{K}) where {L <: AddressMap, K}
    ctx = Score(sel)
    addr = gensym()
    ret = ctx(fn, addr, d)
    b, missed = compare(sel.query, ctx.visited)
    b || error("ScoreError: did not visit all constraints in selection.\nDid not visit: $(missed).")
    return ret, ctx.weight
end
