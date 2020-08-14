# ------------ Choice sites ------------ #

@inline function (ctx::SimulateContext)(c::typeof(plate), 
                                        addr::T, 
                                        d::Distribution{K},
                                        len::Int) where {T <: Address, K}
    v_ret = Vector{eltype(d)}(undef, len)
    v_cl = Vector{Choice{eltype(d)}}(undef, len)
    sc = 0.0
    for i in 1:len
        visit!(ctx, addr => i)
        s = rand(d)
        score = logpdf(d, s)
        sc += score
        v_ret[i] = s
        v_cl[i] = Choice(score, s)
    end
    sc = sum(map(v_cl) do cl
                 get_score(cl)
             end)
    add_call!(ctx, addr, VectorCallSite{typeof(plate)}(VectorTrace(v_cl), sc, d, len, (), v_ret))
    return v_ret
end

# ------------ Call sites ------------ #

@inline function (ctx::SimulateContext)(c::typeof(plate), 
                                        addr::Address, 
                                        call::Function, 
                                        args::Vector)
    visit!(ctx, addr => 1)
    ps = get_subparameters(ctx, addr)
    len = length(args)
    ret, cl = simulate(ps, call, args[1]...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_cl = Vector{typeof(cl)}(undef, len)
    v_ret[1] = ret
    v_cl[1] = cl
    for i in 2:len
        visit!(ctx, addr => i)
        ret, cl = simulate(ps, call, args[i]...)
        v_ret[i] = ret
        v_cl[i] = cl
    end
    sc = sum(map(v_cl) do cl
                 get_score(cl)
             end)
    add_call!(ctx, addr, VectorCallSite{typeof(plate)}(VectorTrace(v_cl), sc, call, length(args), args, v_ret))
    return v_ret
end

# ------------ Convenience ------------ #

function simulate(c::typeof(plate), fn::Function, args::Vector)
    ctx = SimulateContext()
    addr = gensym()
    ret = ctx(plate, addr, fn, args)
    return ret, get_sub(ctx.tr, addr)
end

function simulate(params::P, c::typeof(plate), fn::Function, args::Vector) where P <: AddressMap
    addr = gensym()
    v_ps = learnables(addr => params)
    ctx = SimulateContext(v_ps)
    ret = ctx(plate, addr, fn, args)
    return ret, get_sub(ctx.tr, addr)
end

function simulate(fn::typeof(plate), d::Distribution{T}, len::Int) where T
    ctx = SimulateContext()
    addr = gensym()
    ret = ctx(plate, addr, d, len)
    return ret, get_sub(ctx.tr, addr)
end

