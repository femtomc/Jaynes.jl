@inline function (ctx::ScoreContext)(c::typeof(plate), 
                                     call::Function, 
                                     args::Vector)
    visit!(ctx, 1)
    ss = get_sub(ctx.target, 1)
    len = length(args)
    ret, w = score(ss, call, args[1]...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_ret[1] = ret
    increment!(ctx, w)
    for i in 2:len
        visit!(ctx, i)
        ss = get_sub(ctx.target, i)
        ret, w = score(ss, call, args[i]...)
        v_ret[i] = ret
        increment!(ctx, w)
    end
    return v_ret
end

@inline function (ctx::ScoreContext)(c::typeof(plate), 
                                     addr::A,
                                     call::Function, 
                                     args::Vector) where A <: Address
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ss = get_sub(ctx.target, addr)
    ret, w = score(ss, ps, plate, call, args)
    increment!(ctx, w)
    return ret
end

# ------------ Convenience ------------ #

function score(sel::L, fn::typeof(plate), call::Function, args::Vector) where L <: AddressMap
    ctx = Score(sel, Empty())
    ret = ctx(fn, call, args)
    b, missed = compare(sel, ctx.visited)
    b || error("ScoreError: did not visit all constraints in selection.\nDid not visit: $(missed).")
    return ret, ctx.weight
end

function score(sel::L, params::P, fn::typeof(plate), call::Function, args::Vector) where {L <: AddressMap, P <: AddressMap}
    ctx = Score(sel, params)
    ret = ctx(fn, call, args)
    b, missed = compare(sel, ctx.visited)
    b || error("ScoreError: did not visit all constraints in selection.\nDid not visit: $(missed).")
    return ret, ctx.weight
end

function score(sel::L, fn::typeof(plate), d::Distribution{K}, len::Int) where {L <: AddressMap, K}
    ctx = Score(sel, Empty())
    ret = ctx(fn, d, len)
    b, missed = compare(sel, ctx.visited)
    b || error("ScoreError: did not visit all constraints in selection.\nDid not visit: $(missed).")
    return ret, ctx.weight
end

function score(sel::L, params::P, fn::typeof(plate), d::Distribution{K}, len::Int) where {L <: AddressMap, P <: AddressMap, K}
    ctx = Score(sel, params)
    ret = ctx(fn, d, len)
    b, missed = compare(sel, ctx.visited)
    b || error("ScoreError: did not visit all constraints in selection.\nDid not visit: $(missed).")
    return ret, ctx.weight
end
