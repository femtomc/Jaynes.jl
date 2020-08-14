# ------------ Call sites ------------ #

@inline function (ctx::ScoreContext)(c::typeof(plate), 
                                     addr::Address, 
                                     call::Function, 
                                     args::Vector)
    visit!(ctx, addr => 1)
    ss = get_subselection(ctx, (addr, 1))
    len = length(args)
    ret, w = score(ss, call, args[1]...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_ret[1] = ret
    increment!(ctx, w)
    for i in 2:len
        visit!(ctx, addr => i)
        ss = get_subselection(ctx, (addr, i))
        ret, w = score(ss, call, args[i]...)
        v_ret[i] = ret
        increment!(ctx, w)
    end
    return v_ret
end

# ------------ Convenience ------------ #

function score(sel::L, fn::typeof(plate), call::Function, args::Vector; params = AddressMap()) where L <: AddressMap
    ctx = Score(sel, params)
    addr = gensym()
    ret = ctx(fn, addr, call, args)
    b, missed = compare(sel.query, ctx.visited)
    b || error("ScoreError: did not visit all constraints in selection.\nDid not visit: $(missed).")
    return ret, ctx.weight
end

function score(sel::L, fn::typeof(plate), d::Distribution{K}, len::Int; params = AddressMap()) where {L <: AddressMap, K}
    addr = gensym()
    v_sel = selection(addr => sel)
    ctx = Score(v_sel, params)
    ret = ctx(fn, addr, d, len)
    b, missed = compare(sel.query, ctx.visited)
    b || error("ScoreError: did not visit all constraints in selection.\nDid not visit: $(missed).")
    return ret, ctx.weight
end

