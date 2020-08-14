# ------------ Choice sites ------------ #

@inline function (ctx::GenerateContext)(c::typeof(plate), 
                                        addr::T, 
                                        d::Distribution{K},
                                        len::Int) where {T <: Address, K}
    v_ret = Vector{eltype(d)}(undef, len)
    v_cs = Vector{Choice{eltype(d)}}(undef, len)
    ss = get_sub(ctx.target, addr)
    for i in 1:len
        visit!(ctx, addr => i)
        if haskey(ss, i)
            s = get_sub(ss, i)
            score = logpdf(d, s)
            cs = Choice(score, s)
            increment!(ctx, score)
        else
            s = rand(d)
            score = logpdf(d, s)
            cs = Choice(score, s)
        end
        v_ret[i] = s
        v_cs[i] = cs
    end
    sc = sum(map(v_cs) do cs
                 get_score(cs)
             end)
    add_call!(ctx, addr, VectorCallSite{typeof(plate)}(VectorTrace(v_cs), sc, d, len, (), v_ret))
    return v_ret
end

# ------------ Call sites ------------ #

@inline function (ctx::GenerateContext)(c::typeof(plate), 
                                        addr::Address, 
                                        call::Function, 
                                        args::Vector)
    visit!(ctx, addr => 1)
    len = length(args)
    ss = get_sub(ctx.target, (addr, 1))
    ret, cl, w = generate(ss, call, args[1]...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_cl = Vector{typeof(cl)}(undef, len)
    v_ret[1] = ret
    v_cl[1] = cl
    increment!(ctx, w)
    for i in 2:len
        visit!(ctx, addr => i)
        ss = get_sub(ctx.target, (addr, i))
        ret, cl, w = generate(ss, call, args[i]...)
        v_ret[i] = ret
        v_cl[i] = cl
        increment!(ctx, w)
    end
    sc = sum(map(v_cl) do cl
                 get_score(cl)
             end)
    add_call!(ctx, addr, VectorCallSite{typeof(plate)}(VectorTrace(v_cl), sc, call, len, args, v_ret))
    return v_ret
end

# ------------ Convenience ------------ #

function generate(target::L, fn::typeof(plate), call::Function, args::Vector) where L <: AddressMap
    addr = gensym()
    v_target = schemaion(addr => schema)
    ctx = Generate(v_target)
    ret = ctx(fn, addr, call, args)
    return ret, get_sub(ctx.tr, addr), ctx.weight
end

function generate(target::L, params, fn::typeof(plate), call::Function, args::Vector) where L <: AddressMap
    addr = gensym()
    v_target = schemaion(addr => schema)
    v_ps = learnables(addr => params)
    ctx = Generate(v_target, v_ps)
    ret = ctx(fn, addr, call, args)
    return ret, get_sub(ctx.tr, addr), ctx.weight
end

function generate(target::L, fn::typeof(plate), d::Distribution{K}, len::Int) where {L <: AddressMap, K}
    addr = gensym()
    v_target = schemaion(addr => schema)
    ctx = Generate(v_target)
    ret = ctx(fn, addr, d, len)
    return ret, get_sub(ctx.tr, addr), ctx.weight
end
