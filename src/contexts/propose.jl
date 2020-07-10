mutable struct ProposeContext{T <: Trace} <: ExecutionContext
    tr::T
    weight::Float64
    params::LearnableParameters
    ProposeContext(tr::T) where T <: Trace = new{T}(tr, 0.0, LearnableParameters())
end
Propose() = ProposeContext(Trace())

# ------------ Choice sites ------------ #

@inline function (ctx::ProposeContext)(call::typeof(rand), 
                                       addr::T, 
                                       d::Distribution{K}) where {T <: Address, K}
    s = rand(d)
    score = logpdf(d, s)
    add_choice!(ctx.tr, addr, ChoiceSite(score, s))
    increment!(ctx, score)
    return s
end

# ------------ Call sites ------------ #

@inline function (ctx::ProposeContext)(c::typeof(rand),
                                        addr::T,
                                        call::Function,
                                        args...) where T <: Address
    ret, cl, w = propose(call, args...)
    add_call!(ctx.tr, addr, cl)
    increment!(ctx, w)
    return ret
end

@inline function (ctx::ProposeContext)(c::typeof(foldr), 
                                        fn::typeof(rand), 
                                        addr::Address, 
                                        call::Function, 
                                        len::Int, 
                                        args...)
    p_ctx = Propose()
    ret = p_ctx(call, args...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_tr = Vector{HierarchicalTrace}(undef, len)
    v_ret[1] = ret
    v_tr[1] = p_ctx.tr
    for i in 2:len
        p_ctx.tr = Trace()
        ret = p_ctx(call, v_ret[i-1]...)
        v_ret[i] = ret
        v_tr[i] = p_ctx.tr
    end
    sc = sum(map(v_tr) do tr
                    score(tr)
                end)
    add_call!(ctx.tr, addr, VectorizedCallSite{typeof(foldr)}(v_tr, sc, call, args, v_ret))
    return v_ret
end

@inline function (ctx::ProposeContext)(c::typeof(map), 
                                        fn::typeof(rand), 
                                        addr::Address, 
                                        call::Function, 
                                        args::Vector)
    p_ctx = Propose()
    ret = p_ctx(call, args[1]...)
    len = length(args)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_tr = Vector{HierarchicalTrace}(undef, len)
    v_ret[1] = ret
    v_tr[1] = p_ctx.tr
    for i in 2:len
        n_tr = Trace()
        ret = p_ctx(call, args[i]...)
        v_ret[i] = ret
        v_tr[i] = p_ctx.tr
    end
    sc = sum(map(v_tr) do tr
                    score(tr)
                end)
    add_call!(ctx.tr, addr, VectorizedCallSite{typeof(map)}(v_tr, sc, call, args, v_ret))
    return v_ret
end

# Convenience.
function propose(fn::Function, args...)
    ctx = Propose()
    ret = ctx(fn, args...)
    return ret, BlackBoxCallSite(ctx.tr, fn, args, ret), ctx.weight
end
