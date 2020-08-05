# ------------ Choice sites ------------ #

@inline function (ctx::ParameterBackpropagateContext)(call::typeof(rand), 
                                                      addr::T, 
                                                      d::Distribution{K}) where {T <: Address, K}
    s = get_top(ctx.tr, addr).val
    ctx.weight += logpdf(d, s)
    return s
end

@inline function (ctx::ChoiceBackpropagateContext)(call::typeof(rand), 
                                                   addr::T, 
                                                   d::Distribution{K}) where {T <: Address, K}
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

# ------------ Fillable ------------ #

@inline function (ctx::ParameterBackpropagateContext)(fn::typeof(fillable), addr::Address)
    has_top(ctx.fixed, addr) && return get_top(ctx.fixed, addr)
    error("(fillable): parameter not provided at address $addr.")
end

@inline function (ctx::ChoiceBackpropagateContext)(fn::typeof(fillable), addr::Address)
    has_top(ctx.select, addr) && return get_top(ctx.select, addr)
    error("(fillable): parameter not provided at address $addr.")
end

# ------------ Call sites ------------ #

@inline function (ctx::ParameterBackpropagateContext)(c::typeof(rand),
                                                      addr::T,
                                                      call::Function,
                                                      args...) where T <: Address
    cl = get_sub(ctx.tr, addr)
    ss = get_sub(ctx.fixed, addr)
    param_grads = Gradients()
    ps = get_sub(ctx.initial_params, addr)
    ret = simulate_call_pullback(ss, ps, param_grads, cl, args)
    ctx.param_grads.tree[addr] = param_grads
    return ret
end

@inline function (ctx::ChoiceBackpropagateContext)(c::typeof(rand),
                                                   addr::T,
                                                   call::Function,
                                                   args...) where T <: Address
    cl = get_sub(ctx.tr, addr)
    choice_grads = Gradients()
    ps = get_sub(ctx.initial_params, addr)
    ret = simulate_choice_pullback(ps, choice_grads, get_sub(ctx.select, addr), cl, args)
    ctx.choice_grads.tree[addr] = choice_grads
    return ret
end
