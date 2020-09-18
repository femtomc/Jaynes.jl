import Base: +, setindex!

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

# ------------ includes ------------ #

include("dynamic/backpropagate.jl")
include("factor/backpropagate.jl")

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
