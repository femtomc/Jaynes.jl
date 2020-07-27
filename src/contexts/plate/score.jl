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
