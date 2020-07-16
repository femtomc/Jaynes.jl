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
    increment!(ctx, w)
    for i in 2:len
        visit!(ctx, addr => i)
        ret, cl, w = propose(call, v_ret[i-1]...)
        v_ret[i] = ret
        v_cl[i] = cl
        increment!(ctx, w)
    end
    sc = sum(map(v_cl) do cl
                 get_score(cl)
                end)
    add_call!(ctx, addr, VectorizedSite{typeof(markov)}(VectorizedTrace(v_cl), sc, call, args, v_ret))
    return v_ret
end
