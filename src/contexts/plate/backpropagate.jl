@inline function (ctx::ParameterBackpropagateContext)(c::typeof(plate),
                                                      addr::T,
                                                      call::Function,
                                                      args::Vector) where T <: Address
    param_grads = Gradients()
    ret = simulate_call_pullback(ctx.initial_params, 
                                 param_grads, 
                                 ctx.call, 
                                 args)
    ctx.param_grads.tree[addr] = param_grads
    return ret
end

@inline function (ctx::ParameterBackpropagateContext)(c::typeof(plate),
                                                      call::Function,
                                                      args::Vector) where T <: Address
    param_grads = Gradients()
    ret = simulate_call_pullback(ctx.initial_params, 
                                 param_grads, ctx.call, 
                                 args)
    ctx.param_grads.tree[addr] = param_grads
    return ret
end

@inline function (ctx::ChoiceBackpropagateContext)(c::typeof(plate),
                                                   addr::T,
                                                   call::Function,
                                                   args::Vector) where T <: Address
    choice_grads = Gradients()
    ret = simulate_choice_pullback(ctx.initial_params, 
                                   choice_grads, 
                                   get_sub(ctx.select, addr), 
                                   ctx.call, 
                                   args)
    ctx.choice_grads.tree[addr] = choice_grads
    return ret
end

@inline function (ctx::ChoiceBackpropagateContext)(c::typeof(plate),
                                                   call::Function,
                                                   args::Vector) where T <: Address
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

Zygote.@adjoint function simulate_parameter_pullback(sel, 
                                                     params, 
                                                     param_grads, 
                                                     cl::VectorCallSite{typeof(plate)}, 
                                                     args)
    ret = simulate_parameter_pullback(sel, params, param_grads, cl, args)
    fn = ret_grad -> begin
        arg_grads = accumulate_learnable_gradients!(sel, params, param_grads, get_sub(cl, 1), ret_grad[1])
        for i in 2 : cl.len
            new = accumulate_learnable_gradients!(sel, params, param_grads, get_sub(cl, i), ret_grad[i])
            arg_grads = merge(arg_grads, new)
        end
        (nothing, nothing, nothing, nothing, arg_grads)
    end
    return ret, fn
end

function accumulate_learnable_gradients!(sel, 
                                         initial_params, 
                                         param_grads, 
                                         cl::VectorCallSite{typeof(plate)}, 
                                         ret_grad, 
                                         scaler::Float64 = 1.0) where T <: CallSite
    fn = (args, params) -> begin
        ctx = ParameterBackpropagate(cl, sel, initial_params, params, param_grads)
        ret = ctx(plate, cl.fn, args)
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

Zygote.@adjoint function simulate_choice_pullback(params, 
                                                  choice_grads, 
                                                  choice_selection, 
                                                  cl::VectorCallSite{typeof(plate)}, 
                                                  args)
    ret = simulate_choice_pullback(params, choice_grads, choice_selection, cl, args)
    fn = ret_grad -> begin
        arg_grads = Vector(undef, length(args))
        choice_grads = Dict()
        choice_vals = Dict()
        for i in 1 : cl.len
            arg_grads[i], choice_vals[i], choice_grads[i] = choice_gradients(params, choice_grads, choice_selection, get_sub(cl, i), ret_grad[i])
        end
        (nothing, nothing, nothing, (choice_vals, choice_grads), arg_grads)
    end
    return ret, fn
end

function choice_gradients(initial_params::P, 
                          choice_grads, 
                          choice_selection::K, 
                          cl::VectorCallSite{typeof(plate)}, 
                          ret_grad) where {P <: AddressMap, K <: Target}
    fn = (args, call, sel) -> begin
        ctx = ChoiceBackpropagate(call, sel, initial_params, choice_grads, choice_selection)
        ret = ctx(plate, call.fn, args)
        (ctx.weight, ret)
    end
    (w, r), back = Zygote.pullback(fn, cl.args, cl, selection())
    arg_grads, grad_ref = back((1.0, ret_grad))
    choice_vals = filter!(choice_grads, cl, grad_ref, choice_selection)
    return arg_grads, choice_vals, choice_grads
end
