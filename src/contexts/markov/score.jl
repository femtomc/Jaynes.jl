# ------------ Call sites ------------ #

@inline function (ctx::ScoreContext)(c::typeof(markov), 
                                     addr::Address, 
                                     call::Function, 
                                     len::Int, 
                                     args...)
    visit!(ctx, addr => 1)
    ss = get_sub(ctx.target, addr)
    ret, w = score(get_sub(ss, 1), call, args...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_ret[1] = ret
    increment!(ctx, w)
    for i in 2:len
        visit!(ctx, addr => i)
        ret, w = score(get_sub(ss, i), call, v_ret[i-1]...)
        v_ret[i] = ret
        increment!(ctx, w)
    end
    return v_ret
end

# ------------ Convenience ------------ #

# TODO: fix for dispatch on params.
function score(sel::L, fn::typeof(markov), call::Function, len::Int, args...; params = AddressMap()) where L <: AddressMap
    addr = gensym()
    v_sel = selection(addr => sel)
    ctx = Score(v_sel, params)
    ret = ctx(fn, addr, call, len, args...)
    b, missed = compare(sel.query, ctx.visited)
    b || error("ScoreError: did not visit all constraints in selection.\nDid not visit: $(missed).")
    return ret, ctx.weight
end
