@inline function (ctx::ProposeContext)(c::typeof(plate), 
                                       d::Distribution{K},
                                       len::Int) where {T <: Address, K}
    v_ret = Vector{eltype(d)}(undef, len)
    for i in 1:len
        visit!(ctx, i)
        s = rand(d)
        add_value!(ctx, i, logpdf(d, s), s)
        v_ret[i] = s
    end
    return v_ret
end

@inline function (ctx::ProposeContext)(c::typeof(plate), 
                                       call::Function, 
                                       args::Vector)
    visit!(ctx, 1)
    len = length(args)
    ret, submap, sc = propose(call, args[1]...)
    set_sub!(ctx.map, 1, submap)
    ctx.score += sc
    v_ret = Vector{typeof(ret)}(undef, len)
    v_ret[1] = ret
    for i in 2:len
        visit!(ctx, i)
        ret, submap, sc = propose(call, args[i]...)
        set_sub!(ctx.map, i, submap)
        ctx.score += sc
        v_ret[i] = ret
    end
    return v_ret
end

@inline function (ctx::ProposeContext)(c::typeof(plate), 
                                       addr::A,
                                       call::Function, 
                                       args::Vector) where A <: Address
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ret, submap, sc = propose(ps, plate, call, args)
    set_sub!(ctx.map, addr, submap)
    ctx.score += sc
    return ret
end

# ------------ Convenience ------------ #

function propose(fn::typeof(plate), call::Function, args::Vector)
    ctx = Propose(VectorMap{Value}(length(args)), Empty())
    ret = ctx(fn, call, args)
    return ret, ctx.map, ctx.score
end

function propose(params, fn::typeof(plate), call::Function, args::Vector)
    ctx = Propose(VectorMap{Value}(length(args)), params)
    ret = ctx(fn, call, args)
    return ret, ctx.map, ctx.score
end

function propose(fn::typeof(plate), d::Distribution{K}, len::Int) where K
    ctx = Propose(VectorMap{Value}(length(args)), Empty())
    ret = ctx(fn, d, len)
    return ret, ctx.map, ctx.score
end
