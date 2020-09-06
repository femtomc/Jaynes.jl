# ------------ Choice sites ------------ #

@inline function (ctx::ParameterBackpropagateContext)(call::typeof(rand), 
                                                      addr::T, 
                                                      d::Distribution{K}) where {T <: Address, K}
    s = getindex(ctx.call, addr)
    increment!(ctx, logpdf(d, s))
    return s
end

@inline function (ctx::ChoiceBackpropagateContext)(call::typeof(rand), 
                                                   addr::T, 
                                                   d::Distribution{K}) where {T <: Address, K}
    haskey(ctx.target, addr) || return get_value(get_sub(ctx.call, addr))
    s = read_choice(ctx, addr)
    increment!(ctx, logpdf(d, s))
    return s
end

# ------------ Learnable ------------ #

@inline (ctx::ParameterBackpropagateContext)(fn::typeof(learnable), addr::Address) = read_parameter(ctx, addr)

@inline function (ctx::ChoiceBackpropagateContext)(fn::typeof(learnable), addr::Address)
    haskey(ctx.initial_params, addr) && return getindex(ctx.initial_params, addr)
    error("(learnable): parameter not provided at address $addr.")
end

# ------------ Fillable ------------ #

@inline function (ctx::ParameterBackpropagateContext)(fn::typeof(fillable), addr::Address)
    haskey(ctx.fillables, addr) && return getindex(ctx.fillables, addr)
    error("(fillable): parameter not provided at address $addr.")
end

@inline function (ctx::ChoiceBackpropagateContext)(fn::typeof(fillable), addr::Address)
    haskey(ctx.target, addr) && return getindex(ctx.target, addr)
    error("(fillable): parameter not provided at address $addr.")
end

# ------------ Call sites ------------ #

@inline function (ctx::ParameterBackpropagateContext)(c::typeof(rand), addr::A, call::Function, args...) where A <: Address
    param_grads = Gradients()
    ret = simulate_parameter_pullback(get_sub(ctx.fillables, addr), 
                                      get_sub(ctx.initial_params, addr), 
                                      param_grads, 
                                      get_sub(ctx.call, addr), 
                                      args...)
    ctx.param_grads.tree[addr] = param_grads
    return ret
end

@inline function (ctx::ChoiceBackpropagateContext)(c::typeof(rand), addr::A, call::Function, args...) where A <: Address
    choice_grads = Gradients()
    ret = simulate_choice_pullback(get_sub(ctx.fillables, addr), 
                                   get_sub(ctx.initial_params, addr), 
                                   choice_grads, 
                                   get_sub(ctx.target, addr), 
                                   get_sub(ctx.call, addr), 
                                   args...)
    ctx.choice_grads.tree[addr] = choice_grads
    return ret
end

# ------------ Parameter gradients ------------ #

Zygote.@adjoint function simulate_parameter_pullback(sel, 
                                                     params, 
                                                     param_grads, 
                                                     cl::DynamicCallSite, 
                                                     args...)
    ret = simulate_parameter_pullback(sel, params, param_grads, cl, args...)
    fn = ret_grad -> begin
        arg_grads = accumulate_learnable_gradients!(sel, params, param_grads, cl, ret_grad)
        (nothing, nothing, nothing, nothing, arg_grads...)
    end
    return ret, fn
end

function accumulate_learnable_gradients!(sel, initial_params, param_grads, cl::DynamicCallSite, ret_grad...; scaler::Float64 = 1.0)
    fn = (args, params) -> begin
        ctx = ParameterBackpropagate(cl, sel, initial_params, params, param_grads)
        ret = ctx(cl.fn, args...)
        (ctx.weight, ret...)
    end
    blank = Store()
    _, back = Zygote.pullback(fn, cl.args, blank)
    arg_grads, ps_grad = back((1.0, ret_grad...))
    acc!(param_grads, ps_grad, scaler)
    return arg_grads
end

# ------------ Choice gradients ------------ #

Zygote.@adjoint function simulate_choice_pullback(fillables,
                                                  params, 
                                                  choice_grads, 
                                                  choice_target, 
                                                  cl::DynamicCallSite, 
                                                  args...)
    ret = simulate_choice_pullback(fillables, params, choice_grads, choice_target, cl, args...)
    fn = ret_grad -> begin
        choice_vals, arg_grads = accumulate_choice_gradients!(fillables, params, choice_grads, choice_target, cl, ret_grad...)
        (nothing, nothing, nothing, nothing, (choice_vals, choice_grads), arg_grads...)
    end
    return ret, fn
end

function accumulate_choice_gradients!(fillables::S, initial_params::P, choice_grads, choice_target::K, cl::DynamicCallSite, ret_grad...) where {S <: AddressMap, P <: AddressMap, K <: Target}
    fn = (args, choices) -> begin
        ctx = ChoiceBackpropagate(cl, fillables, initial_params, choices, choice_grads, choice_target)
        ret = ctx(cl.fn, args...)
        (ctx.weight, ret...)
    end
    blank = Store()
    _, back = Zygote.pullback(fn, cl.args, blank)
    arg_grads, grad_ref = back((1.0, ret_grad...))
    choice_vals = filter_acc!(choice_grads, cl, grad_ref, choice_target)
    return choice_vals, arg_grads
end
