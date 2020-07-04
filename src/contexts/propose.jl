mutable struct ProposeContext{T <: Trace} <: ExecutionContext
    tr::T
    ProposeContext(tr::T) where T <: Trace = new{T}(tr)
end
Propose(tr::Trace) = ProposeContext(tr)

# ------------ Choice sites ------------ #

@inline function (ctx::ProposeContext)(call::typeof(rand), 
                                       addr::T, 
                                       d::Distribution{K}) where {T <: Address, K}
    s = rand(d)
    score = logpdf(d, s)
    ctx.tr.chm[addr] = ChoiceSite(score, s)
    ctx.tr.score += score
    return s

end

# ------------ Call sites ------------ #

@inline function (ctx::ProposeContext)(c::typeof(rand),
                                        addr::T,
                                        call::Function,
                                        args...) where T <: Address
    p_ctx = Propose(Trace())
    ret = p_ctx(call, args...)
    ctx.tr.chm[addr] = CallSite(p_ctx.tr, 
                                call, 
                                args, 
                                ret)
    return ret
end

@inline function (ctx::ProposeContext)(c::typeof(foldr), 
                                        fn::typeof(rand), 
                                        addr::Address, 
                                        call::Function, 
                                        len::Int, 
                                        args...)
    p_ctx = Propose(Trace())
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
    tr.chm[addr] = VectorizedCallSite(v_tr, fn, args, v_ret)
    return v_ret
end

@inline function (ctx::ProposeContext)(c::typeof(map), 
                                        fn::typeof(rand), 
                                        addr::Address, 
                                        call::Function, 
                                        args::Vector)
    p_ctx = Propose(Trace())
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
    tr.chm[addr] = VectorizedCallSite(v_tr, fn, args, v_ret)
    return v_ret
end

