# ------------ Call sites ------------ #

@inline function (ctx::ParameterBackpropagateContext)(c::typeof(markov),
                                                      addr::T,
                                                      call::Function,
                                                      len::Int,
                                                      args...) where T <: Address
    param_grads = Gradients()
    ret = simulate_call_pullback(ctx.initial_params, 
                                 param_grads, 
                                 ctx.call, 
                                 args)
    ctx.param_grads.tree[addr] = param_grads
    return ret
end

@inline function (ctx::ParameterBackpropagateContext)(c::typeof(markov),
                                                      call::Function,
                                                      len::Int,
                                                      args...) where T <: Address
    param_grads = Gradients()
    ret = simulate_call_pullback(ctx.initial_params, 
                                 param_grads, 
                                 ctx.call, 
                                 args)
    ctx.param_grads.tree[addr] = param_grads
    return ret
end

@inline function (ctx::ChoiceBackpropagateContext)(c::typeof(markov),
                                                   addr::T,
                                                   call::Function,
                                                   len::Int,
                                                   args...) where T <: Address
    choice_grads = Gradients()
    ret = simulate_choice_pullback(ctx.initial_params, 
                                   choice_grads, 
                                   get_sub(ctx.select, addr), 
                                   ctx.call, 
                                   args)
    ctx.choice_grads.tree[addr] = choice_grads
    return ret
end

@inline function (ctx::ChoiceBackpropagateContext)(c::typeof(markov),
                                                   call::Function,
                                                   len::Int,
                                                   args...) where T <: Address
    choice_grads = Gradients()
    ret = simulate_choice_pullback(ctx.initial_params, 
                                   choice_grads, 
                                   get_sub(ctx.select, addr), 
                                   ctx.call, 
                                   args)
    ctx.choice_grads.tree[addr] = choice_grads
    return ret
end

# ------------ Parameter gradients ------------ #

Zygote.@adjoint function simulate_parameter_pullback(sel, params, param_grads, cl::VectorizedCallSite{typeof(markov)}, args)
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

function accumulate_learnable_gradients!(sel, initial_params, param_grads, cl::VectorizedCallSite{typeof(markov)}, ret_grad, scaler::Float64 = 1.0) where T <: CallSite
    fn = (args, params) -> begin
        ctx = ParameterBackpropagate(cl, sel, initial_params, params, param_grads)
        ret = ctx(markov, cl.fn, cl.len, args...)
        (ctx.weight, ret)
    end
    blank = ParameterStore()
    _, back = Zygote.pullback(fn, cl.args, blank)
    arg_grads, ps_grad = back((1.0, ret_grad))
    if !(ps_grad isa Nothing)
        for (addr, grad) in ps_grad.params
            push!(param_grads, addr, scaler .* grad)
        end
    end
    return arg_grads
end

# ------------ Choice gradients ------------ #
