# ------------ Choice sites ------------ #

@inline function (ctx::ScoreContext)(call::typeof(rand), 
                                     addr::T, 
                                     d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)
    haskey(ctx.target, addr) || error("ScoreError: constrained target must provide constraints for all possible addresses in trace. Missing at address $addr.")
    val = getindex(ctx.target, addr)
    increment!(ctx, logpdf(d, val))
    return val
end

# ------------ Learnable ------------ #

@inline function (ctx::ScoreContext)(fn::typeof(learnable), addr::Address)
    visit!(ctx, addr)
    haskey(ctx.params, addr) && return getindex(ctx.params, addr)
    error("Parameter not provided at address $addr.")
end

# ------------ Fillable ------------ #

@inline function (ctx::ScoreContext)(fn::typeof(fillable), addr::Address)
    haskey(ctx.target, addr) && return getindex(ctx.target, addr)
    error("(fillable): parameter not provided at address $addr.")
end

# ------------ Call sites ------------ #

@inline function (ctx::ScoreContext)(c::typeof(rand),
                                     addr::T,
                                     call::Function,
                                     args...) where T <: Address
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ss = get_sub(ctx.target, addr)
    ret, w = score(ss, ps, call, args...) 
    increment!(ctx, w)
    return ret
end

# ------------ Convenience ------------ #

function score(sel::L, fn::Function, args...) where L <: AddressMap
    ctx = Score(sel)
    ret = ctx(fn, args...)
    b, missed = compare(sel, ctx.visited)
    b || error("ScoreError: did not visit all constraints in target.\nDid not visit: $(missed).")
    return ret, ctx.weight
end

function score(sel::L, params, fn::Function, args...) where L <: AddressMap
    ctx = Score(sel, params)
    ret = ctx(fn, args...)
    b, missed = compare(sel, ctx.visited)
    b || error("ScoreError: did not visit all constraints in target.\nDid not visit: $(missed).")
    return ret, ctx.weight
end

function score(sel::L, fn::typeof(rand), d::Distribution{K}) where {L <: AddressMap, K}
    ctx = Score(sel)
    addr = gensym()
    ret = ctx(fn, addr, d)
    b, missed = compare(sel, ctx.visited)
    b || error("ScoreError: did not visit all constraints in target.\nDid not visit: $(missed).")
    return ret, ctx.weight
end
