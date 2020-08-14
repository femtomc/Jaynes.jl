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
mutable struct ParameterBackpropagateContext{T <: CallSite, S <: ConstrainedSelection} <: BackpropagationContext
    call::T
    weight::Float64
    fixed::S
    initial_params::AddressMap
    params::ParameterStore
    param_grads::Gradients
end

function ParameterBackpropagate(call::T, init, params) where T <: CallSite
    ParameterBackpropagateContext(call, 
                                  0.0, 
                                  selection(), 
                                  init, 
                                  params, 
                                  Gradients())
end

function ParameterBackpropagate(call::T, sel::S, init, params) where {T <: CallSite, S <: ConstrainedSelection}
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
                                  selection(), 
                                  init, 
                                  params, 
                                  param_grads)
end

function ParameterBackpropagate(call::T, sel::S, init, params, param_grads::Gradients) where {T <: CallSite, S <: ConstrainedSelection, K <: Target}
    ParameterBackpropagateContext(call, 
                                  0.0, 
                                  sel, 
                                  init, 
                                  params, 
                                  param_grads)
end

# Choice sites
mutable struct ChoiceBackpropagateContext{T <: CallSite, S <: ConstrainedSelection, K <: Target} <: BackpropagationContext
    call::T
    weight::Float64
    fixed::S
    initial_params::AddressMap
    params::ParameterStore
    choice_grads::Gradients
    select::K
end

function ChoiceBackpropagate(call::T, init, params, choice_grads) where {T <: CallSite, K <: Target}
    ChoiceBackpropagateContext(call, 
                               0.0, 
                               selection(), 
                               init, 
                               params, 
                               choice_grads, 
                               UnconstrainedAllSelection())
end

function ChoiceBackpropagate(call::T, fixed::S, init, params, choice_grads) where {T <: CallSite, S <: ConstrainedSelection, K <: Target}
    ChoiceBackpropagateContext(call, 
                               0.0, 
                               fixed, 
                               init, 
                               params, 
                               choice_grads, 
                               UnconstrainedAllSelection())
end

function ChoiceBackpropagate(call::T, init, params, choice_grads, sel::K) where {T <: CallSite, K <: Target}
    ChoiceBackpropagateContext(call, 
                               0.0, 
                               selection(), 
                               init, 
                               params, 
                               choice_grads, 
                               sel)
end

function ChoiceBackpropagate(call::T, fixed::S, init, params, choice_grads, sel::K) where {T <: CallSite, S <: ConstrainedSelection, K <: Target}
    ChoiceBackpropagateContext(call, 
                               0.0, 
                               fixed, 
                               init, 
                               params, 
                               choice_grads, 
                               sel)
end

# ------------ Learnable ------------ #

read_parameter(ctx::K, addr::Address) where K <: BackpropagationContext = read_parameter(ctx, ctx.params, addr)
read_parameter(ctx::K, params::ParameterStore, addr::Address) where K <: BackpropagationContext = get_top(ctx.initial_params, addr)

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
simulate_parameter_pullback(sel, params, param_grads, cl::T, args) where T <: CallSite = cl.ret

# Grads for choices with differentiable logpdfs.
simulate_choice_pullback(params, choice_grads, choice_selection, cl::T, args) where T <: CallSite = get_ret(cl)

# ------------ filter! choice gradients given selection ------------ #

function filter!(choice_grads, cl::DynamicCallSite, grad_tr::NamedTuple, sel::K) where K <: Target
    values = ConstrainedDynamicSelection()
    for (k, v) in dump_top(cl.trace)
        has_top(sel, k) && begin
            push!(values, k, v.val)
            haskey(grad_tr.trace.choices, k) && push!(choice_grads, k, grad_tr.trace.choices[k].val)
        end
    end
    return values
end

function filter!(choice_grads, cl::DynamicCallSite, grad_tr, sel::K) where K <: Target
    values = ConstrainedDynamicSelection()
    for (k, v) in dump_top(cl.trace)
        has_top(sel, k) && begin
            push!(values, k, v.val)
        end
    end
    return values
end

# ------------ get_choice_gradients ------------ #

function get_choice_gradients(cl::T, ret_grad) where T <: CallSite
    choice_grads = Gradients()
    choice_selection = UnconstrainedAllSelection()
    arg_grads, vals, _ = choice_gradients(AddressMap(), choice_grads, choice_selection, cl, ret_grad)
    return arg_grads, vals, choice_grads
end

function get_choice_gradients(fixed::S, cl::T, ret_grad) where {S <: ConstrainedSelection, T <: CallSite}
    choice_grads = Gradients()
    choice_selection = UnconstrainedAllSelection()
    arg_grads, vals, _ = choice_gradients(fixed, AddressMap(), choice_grads, choice_selection, cl, ret_grad)
    return arg_grads, vals, choice_grads
end

function get_choice_gradients(ps::P, cl::T, ret_grad) where {P <: AddressMap, T <: CallSite}
    choice_grads = Gradients()
    choice_selection = UnconstrainedAllSelection()
    arg_grads, vals, _ = choice_gradients(ps, choice_grads, choice_selection, cl, ret_grad)
    return arg_grads, vals, choice_grads
end

function get_choice_gradients(fixed::S, ps::P, cl::T, ret_grad) where {S <: ConstrainedSelection, P <: AddressMap, T <: CallSite}
    choice_grads = Gradients()
    choice_selection = UnconstrainedAllSelection()
    arg_grads, vals, _ = choice_gradients(fixed, ps, choice_grads, choice_selection, cl, ret_grad)
    return arg_grads, vals, choice_grads
end

function get_choice_gradients(sel::K, cl::T, ret_grad) where {T <: CallSite, K <: Target}
    choice_grads = Gradients()
    arg_grads, vals, _ = choice_gradients(AddressMap(), choice_grads, sel, cl, ret_grad)
    return arg_grads, vals, choice_grads
end

function get_choice_gradients(sel::K, ps::P, cl::T, ret_grad) where {T <: CallSite, K <: Target, P <: AddressMap}
    choice_grads = Gradients()
    arg_grads, vals, _ = choice_gradients(ps, choice_grads, sel, cl, ret_grad)
    return arg_grads, vals, choice_grads
end

function get_choice_gradients(sel::K, fixed::S, cl::T, ret_grad) where {K <: Target, S <: ConstrainedSelection, T <: CallSite}
    choice_grads = Gradients()
    arg_grads, vals, _ = choice_gradients(AddressMap(), choice_grads, sel, cl, ret_grad)
    return arg_grads, vals, choice_grads
end

function get_choice_gradients(sel::K, fixed::S, ps::P, cl::T, ret_grad) where {T <: CallSite, K <: Target, S <: ConstrainedSelection, P <: AddressMap}
    choice_grads = Gradients()
    arg_grads, vals, _ = choice_gradients(ps, choice_grads, sel, cl, ret_grad)
    return arg_grads, vals, choice_grads
end

# ------------ get_learnable_gradients ------------ #

function get_learnable_gradients(ps::P, cl::DynamicCallSite, ret_grad, scaler::Float64 = 1.0) where P <: AddressMap
    param_grads = Gradients()
    arg_grads = accumulate_learnable_gradients!(selection(), ps, param_grads, cl, ret_grad, scaler)
    return arg_grads, param_grads
end

function get_learnable_gradients(sel::K, ps::P, cl::DynamicCallSite, ret_grad, scaler::Float64 = 1.0) where {K <: ConstrainedSelection, P <: AddressMap}
    param_grads = Gradients()
    arg_grads = accumulate_learnable_gradients!(sel, ps, param_grads, cl, ret_grad, scaler)
    return arg_grads, param_grads
end

function get_learnable_gradients(ps::P, cl::VectorizedCallSite, ret_grad, scaler::Float64 = 1.0) where P <: AddressMap
    param_grads = Gradients()
    arg_grads, accumulate_learnable_gradients!(selection(), ps, param_grads, cl, ret_grad, scaler)
    key = keys(param_grads.tree)[1]
    return arg_grads, param_grads[key]
end

function get_learnable_gradients(sel::K, ps::P, cl::VectorizedCallSite, ret_grad, scaler::Float64 = 1.0) where {K <: ConstrainedSelection, P <: AddressMap}
    param_grads = Gradients()
    arg_grads = accumulate_learnable_gradients!(sel, ps, param_grads, cl, ret_grad, scaler)
    key = keys(param_grads.tree)[1]
    return arg_grads, param_grads[key]
end

# ------------ train ------------ #

function train(ps::P, fn::Function, args...; opt = ADAM(0.05, (0.9, 0.8)), iters = 1000) where P <: AddressMap
    for i in 1 : iters
        _, cl = simulate(ps, fn, args...)
        _, grads = get_learnable_gradients(ps, cl, 1.0)
        ps = update_learnables(opt, ps, grads)
    end
    return ps
end

function train(sel::K, ps::P, fn::Function, args...; opt = ADAM(0.05, (0.9, 0.8)), iters = 1000) where {K <: ConstrainedSelection, P <: AddressMap}
    for i in 1 : iters
        _, cl, _ = generate(sel, ps, fn, args...)
        _, grads = get_learnable_gradients(sel, ps, cl, 1.0)
        ps = update_learnables(opt, ps, grads)
    end
    return ps
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
ChoiceBackpropagate(tr::T, init_params, params, choice_grads) where {T <: Trace, K <: Target} = ChoiceBackpropagateContext(tr, 0.0, params, choice_grads, UnconstrainedAllSelection())
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
