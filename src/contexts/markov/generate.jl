@inline function (ctx::GenerateContext)(c::typeof(markov), 
                                        call::Function, 
                                        len::Int, 
                                        args...)
    visit!(ctx, 1)
    ps = get_sub(ctx.params, 1)
    ss = get_sub(ctx.target, 1)
    ret, cl, w = generate(ss, ps, call, args...)
    add_call!(ctx, 1, cl)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_ret[1] = ret
    increment!(ctx, w)
    for i in 2:len
        visit!(ctx, i)
        ps = get_sub(ctx.params, i)
        ss = get_sub(ctx.target, i)
        ret, cl, w = generate(ss, ps, call, v_ret[i-1]...)
        add_call!(ctx, i, cl)
        v_ret[i] = ret
        increment!(ctx, w)
    end
    return v_ret
end

@inline function (ctx::GenerateContext)(c::typeof(markov), 
                                        addr::A,
                                        call::Function, 
                                        len::Int, 
                                        args...) where A <: Address
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ss = get_sub(ctx.target, addr)
    ret, cl, w = generate(ss, ps, markov, call, len, args...)
    add_call!(ctx, addr, cl)
    increment!(ctx, w)
    return ret
end

# ------------ Convenience ------------ #

function generate(target::L, fn::typeof(markov), call::Function, len::Int, args...) where L <: AddressMap
    ctx = Generate(VectorTrace(len), target, Empty())
    ret = ctx(fn, call, len, args...)
    return ret, VectorCallSite{typeof(markov)}(ctx.tr, ctx.score, call, args, ret, len), ctx.weight
end

function generate(target::L, params, fn::typeof(markov), call::Function, len::Int, args...) where L <: AddressMap
    ctx = Generate(VectorTrace(len), target, params)
    ret = ctx(fn, call, len, args...)
    return ret, VectorCallSite{typeof(markov)}(ctx.tr, ctx.score, call, args, ret, len), ctx.weight
end
