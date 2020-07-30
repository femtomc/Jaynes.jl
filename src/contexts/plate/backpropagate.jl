# ------------ Call sites ------------ #

@inline function (ctx::ParameterBackpropagateContext)(c::typeof(plate),
                                                      addr::T,
                                                      call::Function,
                                                      len::Int,
                                                      args...) where T <: Address
    #visit!(ctx.visited, addr)
    vcl = get_call(ctx.tr, addr)
    param_grads = Gradients()
    params = get_sub(ctx.params, addr)
    v_ret = typeof(get_ret(vcl))(undef, len)
    for i in 1 : len
        ret = simulate_call_pullback(params, param_grads, get_call(vcl, i), args)
        ctx.param_grads.tree[addr] += param_grads
        v_ret[i] = ret
    end
    return v_ret
end

@inline function (ctx::ChoiceBackpropagateContext)(c::typeof(plate),
                                                   addr::T,
                                                   call::Function,
                                                   len::Int,
                                                   args...) where T <: Address
    #visit!(ctx.visited, addr)
    vcl = get_call(ctx.tr, addr)
    params = get_sub(ctx.params, addr)
    v_ret = typeof(get_ret(vcl))(undef, len)
    for i in 1 : len
        choice_grads = Gradients()
        ret = simulate_choice_pullback(params, choice_grads, get_sub(ctx.select, addr), get_call(vcl, i), args)
        v_ret[i] = ret
        ctx.choice_grads.tree[addr] += choice_grads
    end
    return v_ret
end
