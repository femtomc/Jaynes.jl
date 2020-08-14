# ------------ Call sites ------------ #

@inline function (ctx::SimulateContext)(c::typeof(markov), 
                                        addr::Address, 
                                        call::Function, 
                                        len::Int, 
                                        args...)
    visit!(ctx, addr => 1)
    ps = get_subparameters(ctx, addr)
    ret, cl = simulate(ps, call, args...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_cl = Vector{typeof(cl)}(undef, len)
    v_ret[1] = ret
    v_cl[1] = cl
    for i in 2:len
        visit!(ctx, addr => i)
        ret, cl = simulate(ps, call, v_ret[i-1]...)
        v_ret[i] = ret
        v_cl[i] = cl
    end
    sc = sum(map(v_cl) do cl
                 get_score(cl)
             end)
    add_call!(ctx, addr, VectorizedCallSite{typeof(markov)}(VectorizedTrace(v_cl), sc, call, len, args, v_ret))
    return v_ret
end

# ------------ Convenience ------------ #

function simulate(c::typeof(markov), fn::Function, len::Int, args...)
    ctx = SimulateContext()
    addr = gensym()
    ret = ctx(markov, addr, fn, len, args...)
    return ret, get_sub(ctx.tr, addr)
end

function simulate(params::P, c::typeof(markov), fn::Function, len::Int, args...) where P <: AddressMap
    addr = gensym()
    v_ps = learnables(addr => params)
    ctx = SimulateContext(v_ps)
    ret = ctx(markov, addr, fn, len, args...)
    return ret, get_sub(ctx.tr, addr)
end
