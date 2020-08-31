@inline function (ctx::SimulateContext)(c::typeof(plate), 
                                        d::Distribution{K},
                                        len::Int) where {A <: Address, K}
    v_ret = Vector{eltype(d)}(undef, len)
    for i in 1 : len
        visit!(ctx, i)
        s = rand(d)
        add_choice!(ctx, i, logpdf(d, s), s)
        v_ret[i] = s
    end
    return v_ret
end

@inline function (ctx::SimulateContext)(c::typeof(plate), 
                                        call::Function, 
                                        args::Vector)
    # First index.
    ps = get_sub(ctx.params, 1)
    len = length(args)
    ret, cl = simulate(ps, call, args[1]...)
    add_call!(ctx, 1, cl)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_ret[1] = ret

    # Rest.
    for i in 2:len
        ps = get_sub(ctx.params, i)
        ret, cl = simulate(ps, call, args[i]...)
        add_call!(ctx, i, cl)
        v_ret[i] = ret
    end
    return v_ret
end

@inline function (ctx::SimulateContext)(c::typeof(plate), 
                                        addr::A,
                                        call::Function, 
                                        args::Vector) where A <: Address
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ret, cl = simulate(ps, plate, call, args)
    add_call!(ctx, addr, cl)
    return ret
end

# ------------ Convenience ------------ #

function simulate(c::typeof(plate), fn::Function, args::Vector)
    ctx = Simulate(VectorTrace(length(args)), Empty())
    ret = ctx(c, fn, args)
    return ret, VectorCallSite{typeof(plate)}(ctx.tr, ctx.score, fn, args, ret, length(args))
end

function simulate(params::P, c::typeof(plate), fn::Function, args::Vector) where P <: AddressMap
    ctx = Simulate(VectorTrace(length(args)), params)
    ret = ctx(c, fn, args)
    return ret, VectorCallSite{typeof(plate)}(ctx.tr, ctx.score, fn, args, ret, length(args))
end

function simulate(c::typeof(plate), d::Distribution{T}, len::Int) where T
    ctx = Simulate(VectorTrace(len), Empty())
    ret = ctx(c, d, len)
    return ret, VectorCallSite{typeof(plate)}(ctx.tr, ctx.score, fn, args, ret, length(args))
end

function simulate(params::P, c::typeof(plate), d::Distribution{T}, len::Int) where {P <: AddressMap, T}
    ctx = Simulate(VectorTrace(len), params)
    ret = ctx(c, d, len)
    return ret, VectorCallSite{typeof(plate)}(ctx.tr, ctx.score, fn, args, ret, length(args))
end
