# ------------ Choice sites ------------ #

@inline function (ctx::ParameterBackpropagateContext)(call::typeof(rand), 
                                                      addr::T, 
                                                      d::Distribution{K}) where {T <: Address, K}
    if haskey(ctx.fixed, addr)
        s = getindex(ctx.fixed, addr)
    else
        s = get_value(get_sub(ctx.call, addr))
    end
    increment!(ctx, logpdf(d, s))
    return s
end

@inline function (ctx::ChoiceBackpropagateContext)(call::typeof(rand), 
                                                   addr::T, 
                                                   d::Distribution{K}) where {T <: Address, K}
    haskey(ctx.target, addr) || return get_value(get_sub(ctx.call, addr))
    s = getindex(ctx.call, addr)
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
    haskey(ctx.fixed, addr) && return getindex(ctx.fixed, addr)
    error("(fillable): parameter not provided at address $addr.")
end

@inline function (ctx::ChoiceBackpropagateContext)(fn::typeof(fillable), addr::Address)
    haskey(ctx.target, addr) && return getindex(ctx.target, addr)
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
    set_sub!(ctx.param_grads, addr, param_grads)
    return ret
end

@inline function (ctx::ChoiceBackpropagateContext)(c::typeof(rand),
                                                   addr::T,
                                                   call::Function,
                                                   args...) where T <: Address
    cl = get_sub(ctx.call, addr)
    ps = get_sub(ctx.initial_params, addr)
    choice_grads = Gradients()
    ret = simulate_choice_pullback(ps, choice_grads, get_sub(ctx.target, addr), cl, args)
    set_sub!(ctx.choice_grads, addr, choice_grads)
    return ret
end

# ------------ Parameter gradients ------------ #

Zygote.@adjoint function simulate_parameter_pullback(sel, params, param_grads, cl::DynamicCallSite, args)
    ret = simulate_parameter_pullback(sel, params, param_grads, cl, args)
    fn = ret_grad -> begin
        arg_grads = accumulate_learnable_gradients!(sel, params, param_grads, cl, ret_grad)
        (nothing, nothing, nothing, nothing, arg_grads)
    end
    return ret, fn
end

function accumulate_learnable_gradients!(sel, initial_params, param_grads, cl::DynamicCallSite, ret_grad, scaler::Float64 = 1.0)
    fn = (args, params) -> begin
        ctx = ParameterBackpropagate(cl, sel, initial_params, params, param_grads)
        ret = ctx(cl.fn, args...)
        (ctx.weight, ret)
    end
    blank = ParameterStore()
    _, back = Zygote.pullback(fn, cl.args, blank)
    arg_grads, ps_grad = back((1.0, ret_grad))
    if !(ps_grad isa Nothing)
        for (addr, grad) in ps_grad.params
            accumulate!(param_grads, addr, scaler .* grad)
        end
    end
    return arg_grads
end

function accumulate_learnable_gradients!(sel, initial_params, param_grads, cl::DynamicCallSite, ret_grad::Tuple, scaler::Float64 = 1.0)
    fn = (args, params) -> begin
        ctx = ParameterBackpropagate(cl, sel, initial_params, params, param_grads)
        ret = ctx(cl.fn, args...)
        (ctx.weight, ret)
    end
    blank = ParameterStore()
    _, back = Zygote.pullback(fn, cl.args, blank)
    arg_grads, ps_grad = back((1.0, ret_grad...))
    if !(ps_grad isa Nothing)
        for (addr, grad) in ps_grad.params
            accumulate!(param_grads, addr, scaler .* grad)
        end
    end
    return arg_grads
end

# ------------ Choice gradients ------------ #

Zygote.@adjoint function simulate_choice_pullback(params, choice_grads, choice_target, cl::DynamicCallSite, args)
    ret = simulate_choice_pullback(params, choice_grads, choice_target, cl, args)
    fn = ret_grad -> begin
        arg_grads, choice_vals, choice_grads = choice_gradients(params, choice_grads, choice_target, cl, ret_grad)
        (nothing, nothing, nothing, (choice_vals, choice_grads), arg_grads)
    end
    return ret, fn
end

function choice_gradients(initial_params::P, choice_grads, choice_target::K, cl::DynamicCallSite, ret_grad) where {P <: AddressMap, K <: Target}
    fn = (args, call, sel) -> begin
        ctx = ChoiceBackpropagate(call, sel, initial_params, ParameterStore(), choice_grads, choice_target)
        ret = ctx(call.fn, args...)
        (ctx.weight, ret)
    end
    _, back = Zygote.pullback(fn, cl.args, cl, target())
    arg_grads, grad_ref = back((1.0, ret_grad))
    choice_vals = filter!(choice_grads, cl, grad_ref, choice_target)
    return arg_grads, choice_vals, choice_grads
end

function choice_gradients(fixed::S, initial_params::P, choice_grads, choice_target::K, cl::DynamicCallSite, ret_grad) where {S <: AddressMap, P <: AddressMap, K <: Target}
    fn = (args, call, sel) -> begin
        ctx = ChoiceBackpropagate(call, sel, initial_params, ParameterStore(), choice_grads, choice_target)
        ret = ctx(call.fn, args...)
        (ctx.weight, ret)
    end
    _, back = Zygote.pullback(fn, cl.args, cl, fixed)
    arg_grads, grad_ref = back((1.0, ret_grad))
    choice_vals = filter!(choice_grads, cl, grad_ref, choice_target)
    return arg_grads, choice_vals, choice_grads
end
