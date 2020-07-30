# ------------ Choice sites ------------ #

@inline function (ctx::SimulateContext)(c::typeof(plate), 
                                        addr::T, 
                                        d::Distribution{K},
                                        len::Int) where {T <: Address, K}
    v_ret = Vector{eltype(d)}(undef, len)
    v_cl = Vector{ChoiceSite{eltype(d)}}(undef, len)
    for i in 1:len
        visit!(ctx, addr => i)
        s = rand(d)
        v_ret[i] = s
        v_cl[i] = ChoiceSite(logpdf(d, s), s)
    end
    sc = sum(map(v_cl) do cl
                 get_score(cl)
             end)
    add_call!(ctx, addr, VectorizedCallSite{typeof(plate)}(VectorizedTrace(v_cl), sc, d, len, (), v_ret))
    return v_ret
end

# ------------ Call sites ------------ #

@inline function (ctx::SimulateContext)(c::typeof(plate), 
                                        addr::Address, 
                                        call::Function, 
                                        args::Vector)
    visit!(ctx, addr => 1)
    len = length(args)
    ret, cl = simulate(call, args[1]...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_cl = Vector{typeof(cl)}(undef, len)
    v_ret[1] = ret
    v_cl[1] = cl
    for i in 2:len
        visit!(ctx, addr => i)
        ret, cl = simulate(call, args[i]...)
        v_ret[i] = ret
        v_cl[i] = cl
    end
    sc = sum(map(v_cl) do cl
                 get_score(cl)
             end)
    add_call!(ctx, addr, VectorizedCallSite{typeof(plate)}(VectorizedTrace(v_cl), sc, call, length(args), args, v_ret))
    return v_ret
end
