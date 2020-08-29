@inline function (ctx::ScoreContext)(c::typeof(markov), 
                                     call::Function, 
                                     len::Int, 
                                     args...)
    visit!(ctx, 1)
    ss = get_sub(ctx.target, 1)
    ps = get_sub(ctx.params, 1)
    ret, w = score(ss, ps, call, args...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_ret[1] = ret
    increment!(ctx, w)
    for i in 2 : len
        visit!(ctx, i)
        ss = get_sub(ctx.target, i)
        ps = get_sub(ctx.params, i)
        ret, w = score(ss, ps, call, v_ret[i-1]...)
        v_ret[i] = ret
        increment!(ctx, w)
    end
    return v_ret
end

@inline function (ctx::ScoreContext)(c::typeof(markov), 
                                     addr::A,
                                     call::Function, 
                                     len::Int, 
                                     args...) where A <: Address
    visit!(ctx, addr)
    ss = get_sub(ctx.target, addr)
    ps = get_sub(ctx.params, addr)
    ret, w = score(ss, ps, markov, call, len, args...)
    increment!(ctx, w)
    return ret
end

# ------------ Convenience ------------ #

function score(sel::L, fn::typeof(markov), call::Function, len::Int, args...) where L <: AddressMap
    ctx = Score(sel, Empty())
    ret = ctx(fn, call, len, args...)
    b, missed = compare(sel, ctx.visited)
    b || error("ScoreError: did not visit all constraints in selection.\nDid not visit: $(missed).")
    return ret, ctx.weight
end

function score(sel::L, params::P, fn::typeof(markov), call::Function, len::Int, args...) where {L <: AddressMap, P <: AddressMap}
    ctx = Score(sel, params)
    ret = ctx(fn, call, len, args...)
    b, missed = compare(sel, ctx.visited)
    b || error("ScoreError: did not visit all constraints in selection.\nDid not visit: $(missed).")
    return ret, ctx.weight
end
