# TODO.
#@inline function (ctx::ForwardModeContext)(c::typeof(markov),
#                                           d::Distribution{K},
#                                           len::Int) where K
#    v_ret = Vector{eltype(d)}(undef, len)
#    for i in 1:len
#        visit!(ctx, i)
#        s = context_getindex(ctx, ctx.map, i)
#        ctx.weight += logpdf(d, s)
#        v_ret[i] = s
#    end
#    return v_ret
#end

# ------------ Choice sites ------------ #

@inline function (ctx::ForwardModeMarkovContext)(call::typeof(rand), 
                                                 addr::A, 
                                                 d::Distribution{K}) where {A <: Address, K}
    visit!(ctx, addr)
    v = context_getindex(ctx, ctx.map, addr)
    ctx.weight += logpdf(d, v)
    v
end

@inline function (ctx::ForwardModeMarkovContext)(c::typeof(markov),
                                                 call::Function,
                                                 len::Int,
                                                 args...)
    # First index.
    visit!(ctx, 1)
    ss = get_sub(ctx.map, 1)
    ps = get_sub(ctx.params, 1)
    ret, w = markov_forward(1 == ctx.target[1] ? ctx.target[2 : end] : (), 
                            ps, ss, Dual(1.0, 0.0), args...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_ret[1] = ret
    ctx.weight += w

    # Rest.
    for i in 2 : len
        visit!(ctx, i)
        ss = get_sub(ctx.map, i)
        ps = get_sub(ctx.params, i)
        ret, w = markov_forward(i == ctx.target[1] ? ctx.target[2 : end] : (), 
                                ps, ss, Dual(1.0, 0.0), v_ret[i - 1])
        v_ret[i] = ret
        ctx.weight += w
    end
    return v_ret
end

@inline function (ctx::ForwardModeMarkovContext)(c::typeof(markov),
                                                 addr::A,
                                                 call::Function,
                                                 len::Int,
                                                 args...) where A <: Address
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ss = get_sub(ctx.map, addr)
    ret, w = forward(ctx.target[2 : end], ps, ss, Dual(1.0, 0.0))
    ctx.weight += w
    return ret
end
