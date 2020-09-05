@inline function (ctx::ParameterBackpropagateContext)(c::typeof(markov),
                                                      call::Function,
                                                      len::Int,
                                                      args...) where T <: Address
    param_grads = Gradients()
    ret = simulate_parameter_pullback(ctx.initial_params, 
                                      param_grads, 
                                      ctx.call, 
                                      args)
    set_sub!(ctx.param_grads, addr, param_grads)
    return ret
end

@inline function (ctx::ParameterBackpropagateContext)(c::typeof(markov),
                                                      addr::T,
                                                      call::Function,
                                                      len::Int,
                                                      args...) where T <: Address
    param_grads = Gradients()
    ret = simulate_parameter_pullback(get_sub(ctx.initial_params, addr),
                                      param_grads, 
                                      get_sub(ctx.call, addr),
                                      args)
    set_sub!(ctx.param_grads, addr, param_grads)
    return ret
end

@inline function (ctx::ChoiceBackpropagateContext)(c::typeof(markov),
                                                   call::Function,
                                                   len::Int,
                                                   args...) where T <: Address
    choice_grads = Gradients()
    ret = simulate_choice_pullback(ctx.initial_params,
                                   choice_grads, 
                                   ctx.select,
                                   ctx.call, 
                                   args)
    set_sub!(ctx.choice_grads, addr, choice_grads)
    return ret
end

@inline function (ctx::ChoiceBackpropagateContext)(c::typeof(markov),
                                                   addr::T,
                                                   call::Function,
                                                   len::Int,
                                                   args...) where T <: Address
    choice_grads = Gradients()
    ret = simulate_choice_pullback(get_sub(ctx.initial_params, addr),
                                   choice_grads, 
                                   get_sub(ctx.target, addr),
                                   get_sub(ctx.call, addr),
                                   args)
    set_sub!(ctx.choice_grads, addr, choice_grads)
    return ret
end

# ------------ Parameter gradients ------------ #

Zygote.@adjoint function simulate_parameter_pullback(sel, 
                                                     params, 
                                                     param_grads, 
                                                     cl::VectorCallSite{typeof(markov)}, 
                                                     args)
    ret = simulate_parameter_pullback(sel, params, param_grads, cl, args)
    fn = ret_grad -> begin
        arg_grads = accumulate_learnable_gradients!(sel, params, param_grads, get_sub(cl, cl.len), ret_grad)
        for i in (cl.len - 1) : -1 : 1
            arg_grads = accumulate_learnable_gradients!(sel, params, param_grads, get_sub(cl, i), arg_grads)
        end
        (nothing, nothing, nothing, nothing, arg_grads)
    end
    return ret, fn
end

function accumulate_learnable_gradients!(sel, 
                                         initial_params, 
                                         param_grads, 
                                         cl::VectorCallSite{typeof(markov)}, 
                                         ret_grad, 
                                         scaler::Float64 = 1.0) where T <: CallSite
    fn = (args, params) -> begin
        ctx = ParameterBackpropagate(cl, sel, initial_params, params, param_grads)
        ret = ctx(markov, cl.fn, cl.len, args...)
        (ctx.weight, ret)
    end
    blank = ParameterStore()
    _, back = Zygote.pullback(fn, cl.args, blank)
    arg_grads, ps_grad = back((1.0, ret_grad))
    acc!(param_grads, ps_grad)
    return arg_grads
end

# ------------ Choice gradients ------------ #

# TODO.
Zygote.@adjoint function simulate_choice_pullback(params, 
                                                  choice_grads, 
                                                  choice_selection, 
                                                  cl::VectorCallSite{typeof(markov)},
                                                  args)
    ret = simulate_choice_pullback(params, choice_grads, choice_selection, cl, args)
    fn = ret_grad -> begin
        choice_vals = target()
        choice_vals[1], arg_grads, choice_grads[1] = choice_gradients(params, choice_grads, choice_selection, get_sub(cl, 1), ret_grad)
        for i in (cl.len - 1) : -1 : 1
            choice_vals[i], arg_grads, choice_grads[i] = choice_gradients(params, Gradients(), choice_selection, get_sub(cl, i), arg_grads)
        end
        (nothing, nothing, nothing, (choice_vals, choice_grads), arg_grads)
    end
    return ret, fn
end

function choice_gradients(initial_params::P, 
                          choice_grads, 
                          choice_selection::K, 
                          cl::VectorCallSite{typeof(markov)}, 
                          ret_grad) where {P <: AddressMap, K <: Target}
    fn = (args, call, sel) -> begin
        ctx = ChoiceBackpropagate(call, sel, initial_params, choice_grads, choice_selection)
        ret = ctx(markov, call.fn, args)
        (ctx.weight, ret)
    end
    (w, r), back = Zygote.pullback(fn, cl.args, cl, selection())
    arg_grads, grad_ref = back((1.0, ret_grad))
    choice_vals = filter!(choice_grads, cl, grad_ref, choice_selection)
    return arg_grads, choice_vals, choice_grads
end
