# ------------ Call sites ------------ #

@inline function (ctx::ProposeContext)(c::typeof(markov), 
                                        addr::Address, 
                                        call::Function, 
                                        len::Int, 
                                        args...)
    visit!(ctx, addr => 1)
    ret, cl, w = propose(call, args...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_cl = Vector{typeof(cl)}(undef, len)
    v_ret[1] = ret
    v_cl[1] = cl
    for i in 2:len
        visit!(ctx, addr => i)
        ret, cl, w = propose(call, v_ret[i-1]...)
        v_ret[i] = ret
        v_cl[i] = cl
    end
    sc = sum(map(v_cl) do cl
                 get_score(cl)
                end)
    add_call!(ctx, addr, VectorCallSite{typeof(markov)}(VectorTrace(v_cl), sc, call, len, args, v_ret))
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
