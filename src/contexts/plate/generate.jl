@inline function (ctx::GenerateContext)(c::typeof(plate), 
                                        d::Distribution{K},
                                        len::Int) where {T <: Address, K}
    v_ret = Vector{eltype(d)}(undef, len)
    ss = get_sub(ctx.target, addr)
    for i in 1:len
        visit!(ctx, i)
        if haskey(ss, i)
            s = get_sub(ss, i)
            add_choice!(ctx, i, logpdf(d, s), s)
            increment!(ctx, score)
        else
            s = rand(d)
            add_choice!(ctx, i, logpdf(d, s), s)
        end
        v_ret[i] = s
    end
    return v_ret
end

@inline function (ctx::GenerateContext)(c::typeof(plate), 
                                        call::Function, 
                                        args::Vector)
    visit!(ctx, 1)
    len = length(args)
    ss = get_sub(ctx.target, 1)
    ret, cl, w = generate(ss, call, args[1]...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_ret[1] = ret
    add_call!(ctx, 1, cl)
    increment!(ctx, w)
    for i in 2:len
        visit!(ctx, i)
        ss = get_sub(ctx.target, i)
        ret, cl, w = generate(ss, call, args[i]...)
        v_ret[i] = ret
        add_call!(ctx, i, cl)
        increment!(ctx, w)
    end
    return v_ret
end

@inline function (ctx::GenerateContext)(c::typeof(plate), 
                                        addr::Address, 
                                        call,
                                        args...)
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ss = get_sub(ctx.target, addr)
    ret, cl, w = generate(ss, ps, plate, call, args...)
    add_call!(ctx, addr, cl)
    increment!(ctx, w)
    return ret
end

# ------------ Convenience ------------ #

function generate(target::L, fn::typeof(plate), call::Function, args::Vector) where L <: AddressMap
    ctx = Generate(VectorTrace(length(args)), target, Empty())
    ret = ctx(fn, call, args)
    return ret, VectorCallSite{typeof(plate)}(ctx.tr, ctx.score, call, args, ret, length(args)), ctx.weight
end

function generate(target::L, params, fn::typeof(plate), call::Function, args::Vector) where L <: AddressMap
    ctx = Generate(VectorTrace(length(args)), target, params)
    ret = ctx(fn, call, args)
    return ret, VectorCallSite{typeof(plate)}(ctx.tr, ctx.score, call, args, ret, length(args)), ctx.weight
end

function generate(target::L, fn::typeof(plate), d::Distribution{K}, len::Int) where {L <: AddressMap, K}
    ctx = Generate(VectorTrace(length(args)), target, Empty())
    ret = ctx(fn, d, len)
    return ret, VectorCallSite{typeof(plate)}(ctx.tr, ctx.score, d, (len, ), ret, len), ctx.weight
end

function generate(target::L, params, fn::typeof(plate), d::Distribution{K}, len::Int) where {L <: AddressMap, K}
    ctx = Generate(VectorTrace(length(args)), target, params)
    ret = ctx(fn, d, len)
    return ret, VectorCallSite{typeof(plate)}(ctx.tr, ctx.score, d, (len, ), ret, len), ctx.weight
end
