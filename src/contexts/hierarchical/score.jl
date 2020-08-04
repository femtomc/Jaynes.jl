# ------------ Choice sites ------------ #

@inline function (ctx::ScoreContext)(call::typeof(rand), 
                                     addr::T, 
                                     d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)
    has_top(ctx.select, addr) || error("ScoreError: constrained selection must provide constraints for all possible addresses in trace. Missing at address $addr.")
    val = get_top(ctx.select, addr)
    increment!(ctx, logpdf(d, val))
    return val
end

# ------------ Learnable ------------ #

@inline function (ctx::ScoreContext)(fn::typeof(learnable), addr::Address)
    visit!(ctx, addr)
    has_top(ctx.params, addr) && return get_top(ctx.params, addr)
    error("Parameter not provided at address $addr.")
end

# ------------ Call sites ------------ #

@inline function (ctx::ScoreContext)(c::typeof(rand),
                                     addr::T,
                                     call::Function,
                                     args...) where T <: Address
    visit!(ctx, addr)
    ps = get_subparameters(ctx, addr)
    ss = get_subselection(ctx, addr)
    ret, w = score(ss, ps, call, args...) 
    increment!(ctx, w)
    return ret
end
