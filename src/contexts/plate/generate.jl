# ------------ Choice sites ------------ #

@inline function (ctx::GenerateContext)(c::typeof(plate), 
                                        addr::T, 
                                        d::Distribution{K},
                                        len::Int) where {T <: Address, K}
    v_ret = Vector{eltype(d)}(undef, len)
    v_cs = Vector{ChoiceSite{eltype(d)}}(undef, len)
    ss = get_subselection(ctx, addr)
    for i in 1:len
        visit!(ctx, addr => i)
        if has_top(ss, i)
            s = get_top(ss, i)
            score = logpdf(d, s)
            cs = ChoiceSite(score, s)
            increment!(ctx, score)
        else
            s = rand(d)
            score = logpdf(d, s)
            cs = ChoiceSite(score, s)
        end
        v_ret[i] = s
        v_cs[i] = cs
    end
    sc = sum(map(v_cs) do cs
                 get_score(cs)
             end)
    add_call!(ctx, addr, VectorizedCallSite{typeof(plate)}(VectorizedTrace(v_cs), sc, d, len, (), v_ret))
    return v_ret
end

# ------------ Call sites ------------ #

@inline function (ctx::GenerateContext)(c::typeof(plate), 
                                        addr::Address, 
                                        call::Function, 
                                        args::Vector)
    visit!(ctx, addr => 1)
    len = length(args)
    ss = get_subselection(ctx, (addr, 1))
    ret, cl, w = generate(ss, call, args[1]...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_cl = Vector{typeof(cl)}(undef, len)
    v_ret[1] = ret
    v_cl[1] = cl
    increment!(ctx, w)
    for i in 2:len
        visit!(ctx, addr => i)
        ss = get_subselection(ctx, (addr, i))
        ret, cl, w = generate(ss, call, args[i]...)
        v_ret[i] = ret
        v_cl[i] = cl
        increment!(ctx, w)
    end
    sc = sum(map(v_cl) do cl
                 get_score(cl)
             end)
    add_call!(ctx, addr, VectorizedCallSite{typeof(plate)}(VectorizedTrace(v_cl), sc, call, len, args, v_ret))
    return v_ret
end

# ------------ Convenience ------------ #

function generate(schema::L, fn::typeof(plate), call::Function, args::Vector) where L <: AddressMap
    addr = gensym()
    v_schema = schemaion(addr => schema)
    ctx = Generate(v_schema)
    ret = ctx(fn, addr, call, args)
    return ret, get_sub(ctx.tr, addr), ctx.weight
end

function generate(schema::L, params, fn::typeof(plate), call::Function, args::Vector) where L <: AddressMap
    addr = gensym()
    v_schema = schemaion(addr => schema)
    v_ps = learnables(addr => params)
    ctx = Generate(v_schema, v_ps)
    ret = ctx(fn, addr, call, args)
    return ret, get_sub(ctx.tr, addr), ctx.weight
end

function generate(schema::L, fn::typeof(plate), d::Distribution{K}, len::Int) where {L <: AddressMap, K}
    addr = gensym()
    v_schema = schemaion(addr => schema)
    ctx = Generate(v_schema)
    ret = ctx(fn, addr, d, len)
    return ret, get_sub(ctx.tr, addr), ctx.weight
end
