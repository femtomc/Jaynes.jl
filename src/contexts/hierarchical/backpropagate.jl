# ------------ Choice sites ------------ #

@inline function (ctx::ParameterBackpropagateContext)(call::typeof(rand), 
                                                      addr::T, 
                                                      d::Distribution{K}) where {T <: Address, K}
    #visit!(ctx.visited, addr)
    s = get_top(ctx.tr, addr).val
    ctx.weight += logpdf(d, s)
    return s
end

@inline function (ctx::ChoiceBackpropagateContext)(call::typeof(rand), 
                                                   addr::T, 
                                                   d::Distribution{K}) where {T <: Address, K}
    #visit!(ctx.visited, addr)
    s = get_top(ctx.tr, addr).val
    ctx.weight += logpdf(d, s)
    return s
end

# ------------ Learnable ------------ #

@inline function (ctx::ParameterBackpropagateContext)(fn::typeof(learnable), addr::Address)
    return read_parameter(ctx, addr)
end

@inline function (ctx::ChoiceBackpropagateContext)(fn::typeof(learnable), addr::Address)
    return read_parameter(ctx, addr)
end

# ------------ Call sites ------------ #

@inline function (ctx::ParameterBackpropagateContext)(c::typeof(rand),
                                                      addr::T,
                                                      call::Function,
                                                      args...) where T <: Address
    #visit!(ctx.visited, addr)
    cl = get_sub(ctx.tr, addr)
    param_grads = Gradients()
    ps = get_sub(ctx.initial_params, addr)
    ret = simulate_call_pullback(ps, param_grads, cl, args)
    ctx.param_grads.tree[addr] = param_grads
    return ret
end

@inline function (ctx::ChoiceBackpropagateContext)(c::typeof(rand),
                                                   addr::T,
                                                   call::Function,
                                                   args...) where T <: Address
    #visit!(ctx.visited, addr)
    cl = get_sub(ctx.tr, addr)
    choice_grads = Gradients()
    ps = get_sub(ctx.initial_params, addr)
    ret = simulate_choice_pullback(ps, choice_grads, get_sub(ctx.select, addr), cl, args)
    ctx.choice_grads.tree[addr] = choice_grads
    return ret
end
