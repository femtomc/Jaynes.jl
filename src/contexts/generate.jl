mutable struct GenerateContext{T <: Trace, K <: ConstrainedSelection} <: ExecutionContext
    tr::T
    select::K
    visited::VisitedSelection
    GenerateContext(tr::T, select::K) where {T <: Trace, K <: ConstrainedSelection} = new{T, K}(tr, select, VisitedSelection())
end
Generate(tr::Trace, select::ConstrainedSelection) = GenerateContext(tr, select)

# ------------ Choice sites ------------ #

@inline function (ctx::GenerateContext)(call::typeof(rand), 
                                        addr::T, 
                                        d::Distribution{K}) where {T <: Address, K}
    if haskey(ctx.select.query, addr)
        s = ctx.select.query[addr]
        score = logpdf(d, s)
        ctx.tr.chm[addr] = ChoiceSite(score, s)
        ctx.tr.score += score
        push!(ctx.visited, addr)
    else
        s = rand(d)
        ctx.tr.chm[addr] = ChoiceSite(logpdf(d, s), s)
        push!(ctx.visited, addr)
    end
    return s
end

# ------------ Call sites ------------ #

@inline function (ctx::GenerateContext)(c::typeof(rand),
                                        addr::T,
                                        call::Function,
                                        args...) where T <: Address
    cg_ctx = GenerateContext(Trace(), ctx.select[addr])
    ret = cg_ctx(call, args...)
    ctx.tr.chm[addr] = CallSite(cg_ctx.tr,
                                call, 
                                args, 
                                ret)
    ctx.visited.tree[addr] = cg_ctx.visited
    return ret
end

@inline function (ctx::GenerateContext)(c::typeof(foldr), 
                                        fn::typeof(rand), 
                                        addr::Address, 
                                        call::Function, 
                                        len::Int, 
                                        args...)
    ug_ctx = Generate(Trace(), ctx.select[addr])
    ret = ug_ctx(call, args...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_tr = Vector{HierarchicalTrace}(undef, len)
    v_ret[1] = ret
    v_tr[1] = ug_ctx.tr
    for i in 2:len
        ug_ctx.tr = Trace()
        ret = ug_ctx(call, v_ret[i-1]...)
        v_ret[i] = ret
        v_tr[i] = ug_ctx.tr
    end
    score = sum(map(v_tr) do tr
                    score(tr)
                end)
    tr.chm[addr] = VectorizedCallSite(v_tr, score, fn, args, v_ret)
    return v_ret
end

@inline function (ctx::GenerateContext)(c::typeof(map), 
                                        fn::typeof(rand), 
                                        addr::Address, 
                                        call::Function, 
                                        args::Vector)
    ug_ctx = Generate(Trace(), ctx.select[addr])
    ret = ug_ctx(call, args[1]...)
    len = length(args)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_tr = Vector{HierarchicalTrace}(undef, len)
    v_ret[1] = ret
    v_tr[1] = ug_ctx.tr
    for i in 2:len
        n_tr = Trace()
        ret = ug_ctx(call, args[i]...)
        v_ret[i] = ret
        v_tr[i] = ug_ctx.tr
    end
    score = sum(map(v_tr) do tr
                    score(tr)
                end)
    tr.chm[addr] = VectorizedCallSite(v_tr, score, fn, args, v_ret)
    return v_ret
end
