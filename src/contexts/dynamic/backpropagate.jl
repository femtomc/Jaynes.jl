# ------------ Choice sites ------------ #

@inline function (ctx::ParameterBackpropagateContext)(call::typeof(rand), 
                                                      addr::T, 
                                                      d::Distribution{K}) where {T <: Address, K}
    if has_top(ctx.fixed, addr)
        s = get_top(ctx.fixed, addr)
    else
        s = get_top(ctx.call, addr).val
    end
    increment!(ctx, logpdf(d, s))
    return s
end

@inline function (ctx::ChoiceBackpropagateContext)(call::typeof(rand), 
                                                   addr::T, 
                                                   d::Distribution{K}) where {T <: Address, K}
    has_top(ctx.select, addr) || return get_top(ctx.call, addr).val
    if has_top(ctx.fixed, addr)
        s = get_top(ctx.fixed, addr)
    else
        s = get_top(ctx.call, addr).val
    end
    increment!(ctx, logpdf(d, s))
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
    cl = get_sub(ctx.call, addr)
    ss = get_sub(ctx.fixed, addr)
    ps = get_sub(ctx.initial_params, addr)
    param_grads = Gradients()
    ret = simulate_call_pullback(ss, ps, param_grads, cl, args)
    ctx.param_grads.tree[addr] = param_grads
    return ret
end

@inline function (ctx::ChoiceBackpropagateContext)(c::typeof(rand),
                                                   addr::T,
                                                   call::Function,
                                                   args...) where T <: Address
    cl = get_sub(ctx.call, addr)
    ps = get_sub(ctx.initial_params, addr)
    choice_grads = Gradients()
    ret = simulate_choice_pullback(ps, choice_grads, get_sub(ctx.select, addr), cl, args)
    ctx.choice_grads.tree[addr] = choice_grads
    return ret
end
