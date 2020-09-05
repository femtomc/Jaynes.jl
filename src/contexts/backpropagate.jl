import Base: +, setindex!, filter!

# Utility.
merge(tp1::Tuple{}, tp2::Tuple{}) = tp1
merge(tp1::Tuple{Nothing}, tp2::Tuple{Nothing}) where T = tp1
merge(tp1::NTuple{N, Float64}, tp2::NTuple{N, Float64}) where N = [tp1[i] + tp2[i] for i in 1 : N]
merge(tp1::Array{Float64}, tp2::NTuple{N, Float64}) where N = [tp1[i] + tp2[i] for i in 1 : N]

# ------------ Parameter store ------------ #

struct ParameterStore
    params::Dict{Address,Any}
    ParameterStore() = new(Dict{Address, Any}())
    ParameterStore(d::Dict{Address, Any}) = new(d)
end
haskey(ps::ParameterStore, addr) = haskey(ps.params, addr)
setindex!(ps::ParameterStore, val, addr) = ps.params[addr] = val

Zygote.@adjoint ParameterStore(params) = ParameterStore(params), store_grad -> (nothing,)

function +(a::ParameterStore, b::ParameterStore)
    params = Dict{Address, Any}()
    for (k, v) in Iterators.flatten((a.params, b.params))
        if !haskey(params, k)
            params[k] = v
        else
            params[k] += v
        end
    end
    ParameterStore(params)
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
    params::ParameterStore
    param_grads::Gradients
end

function ParameterBackpropagate(call::T, init, params) where T <: CallSite
    ParameterBackpropagateContext(call, 
                                  0.0, 
                                  target(), 
                                  init, 
                                  params, 
                                  Gradients())
end

function ParameterBackpropagate(call::T, sel::S, init, params) where {T <: CallSite, S <: AddressMap}
    ParameterBackpropagateContext(call, 
                                  0.0, 
                                  sel, 
                                  init, 
                                  params, 
                                  Gradients())
end

function ParameterBackpropagate(call::T, init, params, param_grads::Gradients) where {T <: CallSite, K <: Target}
    ParameterBackpropagateContext(call, 
                                  0.0, 
                                  target(), 
                                  init, 
                                  params, 
                                  param_grads)
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
    choice_grads::Gradients
    target::K
end

function ChoiceBackpropagate(call::T, init, choice_grads) where {T <: CallSite, K <: Target}
    ChoiceBackpropagateContext(call, 
                               0.0, 
                               target(), 
                               init, 
                               params, 
                               choice_grads, 
                               SelectAll())
end

function ChoiceBackpropagate(call::T, fillables::S, init, choice_grads) where {T <: CallSite, S <: AddressMap, K <: Target}
    ChoiceBackpropagateContext(call, 
                               0.0, 
                               fillables, 
                               init, 
                               choice_grads, 
                               SelectAll())
end

function ChoiceBackpropagate(call::T, init, choice_grads, sel::K) where {T <: CallSite, K <: Target}
    ChoiceBackpropagateContext(call, 
                               0.0, 
                               target(), 
                               init, 
                               choice_grads, 
                               sel)
end

function ChoiceBackpropagate(call::T, fillables::S, init, choice_grads, sel::K) where {T <: CallSite, S <: AddressMap, K <: Target}
    ChoiceBackpropagateContext(call, 
                               0.0, 
                               fillables, 
                               init, 
                               choice_grads, 
                               sel)
end

# ------------ Learnable ------------ #

read_parameter(ctx::K, addr::Address) where K <: BackpropagationContext = read_parameter(ctx, ctx.params, addr)
read_parameter(ctx::K, params::ParameterStore, addr::Address) where K <: BackpropagationContext = getindex(ctx.initial_params, addr)

Zygote.@adjoint function read_parameter(ctx, params, addr)
    ret = read_parameter(ctx, params, addr)
    fn = param_grad -> begin
        state_grad = nothing
        params_grad = ParameterStore(Dict{Address, Any}(addr => param_grad))
        (state_grad, params_grad, nothing)
    end
    return ret, fn
end

# ------------ Parameter sites ------------ #

# Grads for learnable parameters.
simulate_parameter_pullback(sel, params, param_grads, cl::T, args) where T <: CallSite = get_ret(cl)

# Grads for choices with differentiable logpdfs.
simulate_choice_pullback(params, choice_grads, choice_target, cl::T, args) where T <: CallSite = get_ret(cl)

# ------------ filter! choice gradients given target ------------ #

function filter!(choice_grads, cl::DynamicCallSite, grad_tr::NamedTuple, sel::K) where K <: Target
    choices = DynamicMap{Choice}()
    for (k, v) in shallow_iterator(cl)
        haskey(sel, k) && begin
            set_sub!(choices, k, v)
            haskey(grad_tr.trace.tree, k) && accumulate!(choice_grads, k, grad_tr.trace.tree[k].val)
        end
    end
    return choices
end

function filter!(choice_grads, cl::DynamicCallSite, grad_tr, sel::K) where K <: Target
    choices = DynamicMap{Choice}()
    for (k, v) in shallow_iterator(cl)
        haskey(sel, k) && begin
            set_sub!(choices, k, v)
        end
    end
    return choices
end

# ------------ get_choice_gradients ------------ #

function get_choice_gradients(cl::T, ret_grad) where T <: CallSite
    choice_grads = Gradients()
    choice_target = SelectAll()
    arg_grads, vals, _ = choice_gradients(Empty(), choice_grads, choice_target, cl, ret_grad)
    return arg_grads, vals, choice_grads
end

function get_choice_gradients(ps::P, cl::T, ret_grad) where {P <: AddressMap, T <: CallSite}
    choice_grads = Gradients()
    choice_target = SelectAll()
    arg_grads, vals, _ = choice_gradients(ps, choice_grads, choice_target, cl, ret_grad)
    return arg_grads, vals, choice_grads
end

function get_choice_gradients(fillables::S, ps::P, cl::T, ret_grad) where {S <: AddressMap, P <: AddressMap, T <: CallSite}
    choice_grads = Gradients()
    choice_target = SelectAll()
    arg_grads, vals, _ = choice_gradients(fillables, ps, choice_grads, choice_target, cl, ret_grad)
    return arg_grads, vals, choice_grads
end

function get_choice_gradients(sel::K, cl::T, ret_grad) where {T <: CallSite, K <: Target}
    choice_grads = Gradients()
    arg_grads, vals, _ = choice_gradients(Empty(), choice_grads, sel, cl, ret_grad)
    return arg_grads, vals, choice_grads
end

function get_choice_gradients(sel::K, ps::P, cl::T, ret_grad) where {T <: CallSite, K <: Target, P <: AddressMap}
    choice_grads = Gradients()
    arg_grads, vals, _ = choice_gradients(ps, choice_grads, sel, cl, ret_grad)
    return arg_grads, vals, choice_grads
end

function get_choice_gradients(sel::K, fillables::S, ps::P, cl::T, ret_grad) where {T <: CallSite, K <: Target, S <: AddressMap, P <: AddressMap}
    choice_grads = Gradients()
    arg_grads, vals, _ = choice_gradients(ps, choice_grads, sel, cl, ret_grad)
    return arg_grads, vals, choice_grads
end

# ------------ get_learnable_gradients ------------ #

function get_learnable_gradients(ps::P, cl::DynamicCallSite, ret_grad, scaler::Float64 = 1.0) where P <: AddressMap
    param_grads = Gradients()
    arg_grads = accumulate_learnable_gradients!(target(), ps, param_grads, cl, ret_grad, scaler)
    return arg_grads, param_grads
end

function get_learnable_gradients(sel::K, ps::P, cl::DynamicCallSite, ret_grad, scaler::Float64 = 1.0) where {K <: AddressMap, P <: AddressMap}
    param_grads = Gradients()
    arg_grads = accumulate_learnable_gradients!(sel, ps, param_grads, cl, ret_grad, scaler)
    return arg_grads, param_grads
end

function get_learnable_gradients(ps::P, cl::VectorCallSite, ret_grad, scaler::Float64 = 1.0) where P <: AddressMap
    param_grads = Gradients()
    arg_grads = accumulate_learnable_gradients!(target(), ps, param_grads, cl, ret_grad, scaler)
    key = keys(param_grads.tree)[1]
    return arg_grads, param_grads[key]
end

function get_learnable_gradients(sel::K, ps::P, cl::VectorCallSite, ret_grad, scaler::Float64 = 1.0) where {K <: AddressMap, P <: AddressMap}
    param_grads = Gradients()
    arg_grads = accumulate_learnable_gradients!(sel, ps, param_grads, cl, ret_grad, scaler)
    key = keys(param_grads.tree)[1]
    return arg_grads, param_grads[key]
end

# ------------ includes ------------ #

include("dynamic/backpropagate.jl")
include("plate/backpropagate.jl")
include("markov/backpropagate.jl")
include("factor/backpropagate.jl")

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct ParameterBackpropagateContext{T <: Trace} <: BackpropagationContext
    tr::T
    weight::Float64
    initial_params::AddressMap
    params::ParameterStore
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
    params::ParameterStore
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
