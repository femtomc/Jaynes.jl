# ------------ Call sites ------------ #

@inline function (ctx::ScoreContext)(c::typeof(markov), 
                                     addr::Address, 
                                     call::Function, 
                                     len::Int, 
                                     args...)
    visit!(ctx, addr => 1)
    ss = get_subselection(ctx, addr)
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
