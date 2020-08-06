# ------------ Call sites ------------ #

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
