@inline function (ctx::SimulateContext)(c::typeof(markov),
                                        call::Function,
                                        len::Int,
                                        args...)
    visit!(ctx, 1)
    ps = get_sub(ctx.params, 1)
    ret, cl = simulate(ps, call, args...)
    add_call!(ctx, 1, cl)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_ret[1] = ret
    for i in 2:len
        visit!(ctx, i)
        ret, cl = simulate(ps, call, v_ret[i-1]...)
        add_call!(ctx, i, cl)
        v_ret[i] = ret
    end
    return v_ret
end

@inline function (ctx::SimulateContext)(c::typeof(markov),
                                        addr::A,
                                        call::Function,
                                        len::Int,
                                        args...) where A <: Address
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ret, cl = simulate(ps, markov, call, len, args...)
    add_call!(ctx, addr, cl)
    return ret
end

# ------------ Convenience ------------ #

function simulate(c::typeof(markov), fn::Function, len::Int, args...)
    ctx = Simulate(VectorTrace(len), Empty())
    ret = ctx(markov, fn, len, args...)
    return ret, VectorCallSite{typeof(markov)}(ctx.tr, ctx.score, fn, args, ret, len)
end

function simulate(params::P, c::typeof(markov), fn::Function, len::Int, args...) where P <: AddressMap
    ctx = Simulate(VectorTrace(len), params)
    ret = ctx(markov, fn, len, args...)
    return ret, VectorCallSite{typeof(markov)}(ctx.tr, ctx.score, fn, args, ret, len)
end
