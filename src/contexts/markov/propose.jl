@inline function (ctx::ProposeContext)(c::typeof(markov), 
                                       call::Function, 
                                       len::Int, 
                                       args...)
    visit!(ctx, 1)
    ps = get_sub(ctx.params, 1)
    ret, submap, sc = propose(ps, call, args...)
    set_sub!(ctx.map, 1, submap)
    ctx.score += sc
    for i in 2:len
        visit!(ctx, i)
        ps = get_sub(ctx.params, i)
        ret, submap, sc = propose(ps, call, ret...)
        set_sub!(ctx.map, i, submap)
        ctx.score += sc
    end
    return ret
end

@inline function (ctx::ProposeContext)(c::typeof(markov), 
                                       addr::A, 
                                       call::Function, 
                                       len::Int, 
                                       args...) where A <: Address
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ret, submap, sc = propose(ps, markov, call, len, args...)
    set_sub!(ctx.map, addr, submap)
    ctx.score += sc
    return ret
end

# ------------ Convenience ------------ #

function propose(fn::typeof(markov), call::Function, len::Int, args...)
    ctx = Propose(VectorMap{Value}(len), Empty())
    ret = ctx(fn, call, len, args...)
    return ret, ctx.map, ctx.score
end

function propose(params, fn::typeof(markov), call::Function, len::Int, args...)
    ctx = Propose(VectorMap{Value}(len), params)
    ret = ctx(fn, call, len, args...)
    return ret, ctx.map, ctx.score
end
