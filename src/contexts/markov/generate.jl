# ------------ Call sites ------------ #

@inline function (ctx::GenerateContext)(c::typeof(markov), 
                                        addr::Address, 
                                        call::Function, 
                                        len::Int, 
                                        args...)
    if len == 0
        return Vector()
    else
        visit!(ctx, addr => 1)
        ss = get_subselection(ctx, (addr, 1))
        ret, cl, w = generate(ss, call, args...)
        v_ret = Vector{typeof(ret)}(undef, len)
        v_cl = Vector{typeof(cl)}(undef, len)
        v_ret[1] = ret
        v_cl[1] = cl
        increment!(ctx, w)
        for i in 2:len
            visit!(ctx, addr => i)
            ss = get_subselection(ctx, (addr, i))
            ret, cl, w = generate(ss, call, v_ret[i-1]...)
            v_ret[i] = ret
            v_cl[i] = cl
            increment!(ctx, w)
        end
        sc = sum(map(v_cl) do cl
                     get_score(cl)
                 end)
        add_call!(ctx, addr, VectorizedCallSite{typeof(markov)}(VectorizedTrace(v_cl), sc, call, (len, args...), v_ret))
        return v_ret
    end
end
