# Utility.
merge(tp1::Tuple{}, tp2::Tuple{}) = tp1
merge(tp1::Tuple{Nothing}, tp2::Tuple{Nothing}) where T = tp1
merge(tp1::NTuple{N, Float64}, tp2::NTuple{N, Float64}) where N = [tp1[i] + tp2[i] for i in 1 : N]
merge(tp1::Array{Float64}, tp2::NTuple{N, Float64}) where N = [tp1[i] + tp2[i] for i in 1 : N]

# ------------ Gradient store ------------ #

struct Store
    params::Dict{Address,Any}
    Store() = new(Dict{Address, Any}())
    Store(d::Dict{Address, Any}) = new(d)
end
haskey(ps::Store, addr) = haskey(ps.params, addr)
setindex!(ps::Store, val, addr) = ps.params[addr] = val
getindex(ps::Store, addr) = ps.params[addr]

Zygote.@adjoint Store(params) = Store(params), store_grad -> (nothing,)

function +(a::Store, b::Store)
    params = Dict{Address, Any}()
    for (k, v) in Iterators.flatten((a.params, b.params))
        if !haskey(params, k)
            params[k] = v
        else
            params[k] += v
        end
    end
    Store(params)
end

# ------------ Backpropagation contexts ------------ #

abstract type BackpropagationContext <: ExecutionContext end

# Go go dynamo!
@dynamo function (bx::BackpropagationContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    ir = recur(ir)
    jaynesize_transform!(ir)
    ir
end
(bx::BackpropagationContext)(::typeof(Core._apply_iterate), f, c::typeof(trace), args...) = bx(c, flatten(args)...)
function (bx::BackpropagationContext)(::typeof(Base.collect), generator::Base.Generator)
    map(generator.iter) do i
        bx(generator.f, i)
    end
end

# Learnable parameters
mutable struct ParameterBackpropagateContext{T <: CallSite, 
                                             S <: AddressMap,
                                             P <: AddressMap} <: BackpropagationContext
    call::T
    weight::Float64
    fillables::S
    initial_params::P
    params::Store
    param_grads::Gradients
end

function ParameterBackpropagate(call::T, sel::S, init, params, param_grads::Gradients) where {T <: CallSite, S <: AddressMap, K <: Target}
    ParameterBackpropagateContext(call, 
                                  0.0, 
                                  sel, 
                                  init, 
                                  params, 
                                  param_grads)
end

# Choice sites
mutable struct ChoiceBackpropagateContext{T <: CallSite, 
                                          S <: AddressMap, 
                                          P <: AddressMap, 
                                          K <: Target} <: BackpropagationContext
    call::T
    weight::Float64
    fillables::S
    initial_params::P
    choices::Store
    choice_grads::Gradients
    target::K
end

function ChoiceBackpropagate(call::T, fillables::S, init, choice_store, choice_grads, sel::K) where {T <: CallSite, S <: AddressMap, K <: Target}
    ChoiceBackpropagateContext(call, 
                               0.0, 
                               fillables, 
                               init, 
                               choice_store,
                               choice_grads,
                               sel)
end

# ------------ Reading parameters and choices ------------ #

read_parameter(ctx::K, addr::Address) where K <: BackpropagationContext = read_parameter(ctx, ctx.params, addr)
read_parameter(ctx::K, params::Store, addr::Address) where K <: BackpropagationContext = getindex(ctx.initial_params, addr)

Zygote.@adjoint function read_parameter(ctx, params, addr)
    ret = read_parameter(ctx, params, addr)
    fn = param_grad -> begin
        state_grad = nothing
        params_grad = Store(Dict{Address, Any}(addr => param_grad))
        (state_grad, params_grad, nothing)
    end
    return ret, fn
end

read_choice(ctx::K, addr::Address) where K <: BackpropagationContext = read_choice(ctx, ctx.choices, addr)
read_choice(ctx::K, choices, addr::Address) where K <: BackpropagationContext = getindex(ctx.call, addr)

Zygote.@adjoint function read_choice(ctx, call, addr)
    ret = read_choice(ctx, call, addr)
    fn = choice_grad -> begin
        state_grad = nothing
        choices_grad = Store(Dict{Address, Any}(addr => choice_grad))
        (state_grad, choices_grad, nothing)
    end
    return ret, fn
end

# ------------ Simulate function calls ------------ #

# Grads for learnable parameters.
simulate_parameter_pullback(sel, params, param_grads, cl::T, args...) where T <: CallSite = get_ret(cl)

# Grads for choices with differentiable logpdfs.
simulate_choice_pullback(fillables, params, choice_grads, choice_target, cl::T, args...) where T <: CallSite = get_ret(cl)

# ------------ get_choice_gradients ------------ #

function get_choice_gradients(cl::T, ret_grad...) where T <: CallSite
    choice_grads = Gradients()
    vals, arg_grads = accumulate_choice_gradients!(Empty(), Empty(), choice_grads, SelectAll(), cl, ret_grad...)
    return vals, arg_grads, choice_grads
end

function get_choice_gradients(ps::P, cl::T, ret_grad...) where {P <: AddressMap, T <: CallSite}
    choice_grads = Gradients()
    vals, arg_grads = accumulate_choice_gradients!(Empty(), ps, choice_grads, SelectAll(), cl, ret_grad...)
    return vals, arg_grads, choice_grads
end

function get_choice_gradients(fillables::S, ps::P, cl::T, ret_grad...) where {S <: AddressMap, P <: AddressMap, T <: CallSite}
    choice_grads = Gradients()
    vals, arg_grads = accumulate_choice_gradients!(fillables, ps, choice_grads, SelectAll(), cl, ret_grad...)
    return vals, arg_grads, choice_grads
end

function get_choice_gradients(sel::K, cl::T, ret_grad...) where {T <: CallSite, K <: Target}
    choice_grads = Gradients()
    vals, arg_grads = accumulate_choice_gradients!(Empty(), Empty(), choice_grads, sel, cl, ret_grad...)
    return vals, arg_grads, choice_grads
end

function get_choice_gradients(sel::K, ps::P, cl::T, ret_grad...) where {T <: CallSite, K <: Target, P <: AddressMap}
    choice_grads = Gradients()
    vals, arg_grads = accumulate_choice_gradients!(Empty(), ps, choice_grads, sel, cl, ret_grad...)
    return vals, arg_grads, choice_grads
end

function get_choice_gradients(sel::K, fillables::S, ps::P, cl::T, ret_grad...) where {T <: CallSite, K <: Target, S <: AddressMap, P <: AddressMap}
    choice_grads = Gradients()
    vals, arg_grads = accumulate_choice_gradients!(fillables, ps, choice_grads, sel, cl, ret_grad...)
    return vals, arg_grads, choice_grads
end

# ------------ get_learnable_gradients ------------ #

function get_learnable_gradients(ps::P, cl::DynamicCallSite, ret_grad...; scaler::Float64 = 1.0) where P <: AddressMap
    param_grads = Gradients()
    arg_grads = accumulate_learnable_gradients!(target(), ps, param_grads, cl, ret_grad...; scaler = scaler)
    return arg_grads, param_grads
end

function get_learnable_gradients(sel::K, ps::P, cl::DynamicCallSite, ret_grad...; scaler::Float64 = 1.0) where {K <: AddressMap, P <: AddressMap}
    param_grads = Gradients()
    arg_grads = accumulate_learnable_gradients!(sel, ps, param_grads, cl, ret_grad...; scaler = scaler)
    return arg_grads, param_grads
end

# Convenience utilities (used in implementations of accumulate_learnable_gradients! and accumulate_choice_gradients! for each type of call site).
function acc!(param_grads, ::Nothing, scaler) end
function acc!(param_grads, ps_grad, scaler)
    for (addr, grad) in ps_grad.params
        accumulate!(param_grads, addr, scaler .* grad)
    end
end

function filter_acc!(choice_grads, cl, grad_tr::Store, sel::K) where K <: Target
    _, choices = projection(cl, sel)
    for (k, v) in shallow_iterator(cl)
        k in sel && begin
            haskey(grad_tr, k) && accumulate!(choice_grads, k, grad_tr[k])
        end
    end
    return choices
end

@inline filter_acc!(choice_grads, cl, grad_tr, sel::K) where K <: Target = projection(cl, sel)[2]

# ------------ Choice sites ------------ #

@inline function (ctx::ParameterBackpropagateContext)(call::typeof(trace), 
                                                      addr::T, 
                                                      d::Distribution{K}) where {T <: Address, K}
    s = getindex(ctx.call, addr)
    increment!(ctx, logpdf(d, s))
    return s
end

@inline function (ctx::ChoiceBackpropagateContext)(call::typeof(trace), 
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

@inline function (ctx::ParameterBackpropagateContext)(c::typeof(trace), addr::A, call::Function, args...) where A <: Address
    param_grads = Gradients()
    ret = simulate_parameter_pullback(get_sub(ctx.fillables, addr), 
                                      get_sub(ctx.initial_params, addr), 
                                      param_grads, 
                                      get_sub(ctx.call, addr), 
                                      args)
    ctx.param_grads.tree[addr] = param_grads
    return ret
end

@inline function (ctx::ChoiceBackpropagateContext)(c::typeof(trace), addr::A, call::Function, args...) where A <: Address
    choice_grads = Gradients()
    ret = simulate_choice_pullback(get_sub(ctx.fillables, addr), 
                                   get_sub(ctx.initial_params, addr), 
                                   choice_grads, 
                                   get_sub(ctx.target, addr), 
                                   get_sub(ctx.call, addr), 
                                   args)
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
        arg_grads = accumulate_learnable_gradients!(sel, params, param_grads, cl, ret_grad...)
        (nothing, nothing, nothing, nothing, arg_grads)
    end
    return ret, fn
end

function accumulate_learnable_gradients!(sel, initial_params, param_grads, cl::DynamicCallSite, ret_grad...; scaler::Float64 = 1.0)
    fn = (args, params) -> begin
        ctx = ParameterBackpropagate(cl, sel, initial_params, params, param_grads)
        ret = ctx(cl.fn, args...)
        (ctx.weight, ret)
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
        (nothing, nothing, nothing, nothing, (choice_vals, choice_grads), arg_grads)
    end
    return ret, fn
end

function accumulate_choice_gradients!(fillables::S, initial_params::P, choice_grads, choice_target::K, cl::DynamicCallSite, ret_grad...) where {S <: AddressMap, P <: AddressMap, K <: Target}
    fn = (args, choices) -> begin
        ctx = ChoiceBackpropagate(cl, fillables, initial_params, choices, choice_grads, choice_target)
        ret = ctx(cl.fn, args...)
        (ctx.weight, ret)
    end
    blank = Store()
    _, back = Zygote.pullback(fn, cl.args, blank)
    arg_grads, grad_ref = back((1.0, ret_grad...))
    choice_vals = filter_acc!(choice_grads, cl, grad_ref, choice_target)
    return choice_vals, arg_grads
end

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct ParameterBackpropagateContext{T <: Trace} <: BackpropagationContext
    tr::T
    weight::Float64
    initial_params::AddressMap
    params::Store
    param_grads::Gradients
end
```
`ParameterBackpropagateContext` is used to compute the gradients of parameters with respect to following objective:

Outer constructors:
```julia
ParameterBackpropagate(tr::T, params) where T <: Trace = ParameterBackpropagateContext(tr, 0.0, params, Gradients())
ParameterBackpropagate(tr::T, params, param_grads::Gradients) where {T <: Trace, K <: Target} = ParameterBackpropagateContext(tr, 0.0, params, param_grads)
```
""", ParameterBackpropagateContext)

@doc(
"""
```julia
mutable struct ChoiceBackpropagateContext{T <: Trace} <: BackpropagationContext
    tr::T
    weight::Float64
    initial_params::AddressMap
    params::Store
    param_grads::Gradients
end
```
`ChoiceBackpropagateContext` is used to compute the gradients of choices with respect to following objective:

Outer constructors:
```julia
ChoiceBackpropagate(tr::T, init_params, params, choice_grads) where {T <: Trace, K <: Target} = ChoiceBackpropagateContext(tr, 0.0, params, choice_grads, SelectAll())
ChoiceBackpropagate(tr::T, init_params, params, choice_grads, sel::K) where {T <: Trace, K <: Target} = ChoiceBackpropagateContext(tr, 0.0, params, choice_grads, sel)
```
""", ChoiceBackpropagateContext)

@doc(
"""
```julia
gradients = get_choice_gradients(params, cl::T, ret_grad) where T <: CallSite
gradients = get_choice_gradients(cl::T, ret_grad) where T <: CallSite
```

Returns a `Gradients` object which tracks the gradients with respect to the objective of random choices with differentiable `logpdf` in the program.
""", get_choice_gradients)

@doc(
"""
```julia
gradients = get_learnable_gradients(params, cl::T, ret_grad, scaler::Float64 = 1.0) where T <: CallSite
```

Returns a `Gradients` object which tracks the gradients of the objective with respect to parameters in the program.
""", get_learnable_gradients)
