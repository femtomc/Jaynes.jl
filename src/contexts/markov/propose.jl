# ------------ Call sites ------------ #

@inline function (ctx::ProposeContext)(c::typeof(markov), 
                                       addr::Address, 
                                       call::Function, 
                                       len::Int, 
                                       args...)
    visit!(ctx, addr)
    visit!(ctx, addr => 1)
    ret, submap, sc = propose(call, args...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_submap = Vector{typeof(submap)}(undef, len)
    v_ret[1] = ret
    v_submap[1] = submap
    for i in 2:len
        visit!(ctx, addr => i)
        ret, submap, w = propose(call, v_ret[i-1]...)
        v_ret[i] = ret
        v_submap[i] = submap
        sc += w
    end
    ctx.score += sc
    set_sub!(ctx.map, addr, VectorMap{Value}(v_submap))
    return v_ret
end

# ------------ Convenience ------------ #

function propose(fn::typeof(markov), call::Function, len::Int, args...)
    ctx = Propose()
    addr = gensym()
    ret = ctx(fn, addr, call, len, args...)
    return ret, get_sub(ctx.tr, addr), ctx.score
end

function propose(params, fn::typeof(markov), call::Function, len::Int, args...)
    ctx = Propose(params)
    addr = gensym()
    ret = ctx(fn, addr, call, len, args...)
    return ret, get_sub(ctx.tr, addr), ctx.score
end
