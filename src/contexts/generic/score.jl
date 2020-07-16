# ------------ Choice sites ------------ #

@inline function (ctx::ScoreContext)(call::typeof(rand), 
                                     addr::T, 
                                     d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)
    has_query(ctx.select, addr) || error("ScoreError: constrained selection must provide constraints for all possible addresses in trace. Missing at address $addr.")
    val = get_query(ctx.select, addr)
    increment!(ctx, logpdf(d, val))
    return val
end

# ------------ Call sites ------------ #

@inline function (ctx::ScoreContext)(c::typeof(rand),
                                     addr::T,
                                     call::Function,
                                     args...) where T <: Address
    visit!(ctx, addr)
    ss = get_subselection(ctx, addr)
    ret, w = score(ss, call, args...) 
    increment!(ctx, w)
    return ret
end
