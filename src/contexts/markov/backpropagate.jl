@inline function (ctx::ParameterBackpropagateContext)(c::typeof(markov),
                                                      call::Function,
                                                      len::Int,
                                                      args...) where T <: Address
    ret = simulate_parameter_pullback(ctx.initial_params, 
                                      ctx.param_grads, 
                                      ctx.call, 
                                      args...)
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
                                      args...)
    ctx.param_grads.tree[addr] = param_grads
    return ret
end

@inline function (ctx::ChoiceBackpropagateContext)(c::typeof(markov),
                                                   call::Function,
                                                   len::Int,
                                                   args...) where T <: Address
    ret = simulate_choice_pullback(ctx.fillables,
                                   ctx.initial_params,
                                   ctx.choice_grads, 
                                   ctx.select,
                                   ctx.call, 
                                   args...)
    return ret
end

@inline function (ctx::ChoiceBackpropagateContext)(c::typeof(markov),
                                                   addr::T,
                                                   call::Function,
                                                   len::Int,
                                                   args...) where T <: Address
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
                                                     cl::VectorCallSite{typeof(markov)}, 
                                                     args...)
    ret = simulate_parameter_pullback(sel, params, param_grads, cl, args...)
    fn = ret_grad -> begin
        arg_grads = accumulate_learnable_gradients!(sel, params, param_grads, get_sub(cl, cl.len), ret_grad...)
        for i in (cl.len - 1) : -1 : 1
            arg_grads = accumulate_learnable_gradients!(sel, params, param_grads, get_sub(cl, i), arg_grads...)
        end
        (nothing, nothing, nothing, nothing, arg_grads...)
    end
    return ret, fn
end

function accumulate_learnable_gradients!(sel, 
                                         initial_params, 
                                         param_grads, 
                                         cl::VectorCallSite{typeof(markov)}, 
                                         ret_grad...;
                                         scaler::Float64 = 1.0) where T <: CallSite
    fn = (args, params) -> begin
        ctx = ParameterBackpropagate(cl, sel, initial_params, params, param_grads)
        ret = ctx(markov, cl.fn, cl.len, args...)
        (ctx.weight, ret)
    end
    blank = ParameterStore()
    _, back = Zygote.pullback(fn, cl.args, blank)
    arg_grads, ps_grad = back((1.0, ret_grad...))
    acc!(param_grads, ps_grad)
    return arg_grads
end

# ------------ Choice gradients ------------ #

Zygote.@adjoint function simulate_choice_pullback(fillables,
                                                  params, 
                                                  choice_grads, 
                                                  target, 
                                                  cl::VectorCallSite{typeof(markov)},
                                                  args...)
    ret = simulate_choice_pullback(fillables, params, choice_grads, target, cl, args...)
    fn = ret_grad -> begin
        choice_vals = VectorTrace(cl.len)
        sub_grads = Gradients()
        choice_vals[cl.len], arg_grads = accumulate_choice_gradients!(get_sub(fillables, cl.len), get_sub(params, cl.len), sub_grads, get_sub(target, cl.len), get_sub(cl, cl.len), ret_grad...)
        choice_grads[cl.len] = sub_grads
        for i in (cl.len - 1) : -1 : 1
            sub_grads = Gradients()
            choice_vals[i], arg_grads = accumulate_choice_gradients!(get_sub(fillables, i), get_sub(params, i), sub_grads, get_sub(target, i), get_sub(cl, i), arg_grads...)
            choice_grads[i] = sub_grads
        end
        (nothing, nothing, nothing, nothing, (choice_vals, choice_grads), arg_grads...)
    end
    return ret, fn
end

function accumulate_choice_gradients!(fillables::S,
                                      initial_params::P, 
                                      choice_grads, 
                                      choice_target::K, 
                                      cl::VectorCallSite{typeof(markov)}, 
                                      ret_grad...) where {S <: AddressMap, P <: AddressMap, K <: Target}
    fn = (args, choices) -> begin
        ctx = ChoiceBackpropagate(cl, fillables, initial_params, choices, choice_grads, choice_target)
        ret = ctx(markov, call.fn, args)
        (ctx.weight, ret)
    end
    blank = Store()
    _, back = Zygote.pullback(fn, cl.args, blank)
    arg_grads, grad_ref = back((1.0, ret_grad...))
    choice_vals = filter_acc!(choice_grads, cl, grad_ref, target)
    return choice_vals, arg_grads
end
