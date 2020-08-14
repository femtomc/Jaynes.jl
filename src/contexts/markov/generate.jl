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
        ps = get_sub(ctx.params, addr)
        ss = get_sub(ctx.schema, (addr, 1))
        ret, cl, w = generate(ss, ps, call, args...)
        v_ret = Vector{typeof(ret)}(undef, len)
        v_cl = Vector{typeof(cl)}(undef, len)
        v_ret[1] = ret
        v_cl[1] = cl
        increment!(ctx, w)
        for i in 2:len
            visit!(ctx, addr => i)
            ss = get_sub(ctx.schema, (addr, i))
            ret, cl, w = generate(ss, ps, call, v_ret[i-1]...)
            v_ret[i] = ret
            v_cl[i] = cl
            increment!(ctx, w)
        end
        sc = sum(map(v_cl) do cl
                     get_score(cl)
                 end)
        add_call!(ctx, addr, VectorizedCallSite{typeof(markov)}(VectorizedTrace(v_cl), sc, call, len, args, v_ret))
        return v_ret
    end
end

# ------------ Convenience ------------ #

function generate(schema::L, fn::typeof(markov), call::Function, len::Int, args...) where L <: AddressMap
    addr = gensym()
    v_schema = schemaion(addr => schema)
    ctx = Generate(v_schema)
    ret = ctx(fn, addr, call, len, args...)
    return ret, get_sub(ctx.tr, addr), ctx.weight
end

function generate(schema::L, params, fn::typeof(markov), call::Function, len::Int, args...) where L <: AddressMap
    addr = gensym()
    v_schema = schemaion(addr => schema)
    v_ps = learnables(addr => params)
    ctx = Generate(v_schema, v_ps)
    ret = ctx(fn, addr, call, len, args...)
    return ret, get_sub(ctx.tr, addr), ctx.weight
end

