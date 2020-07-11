import Base: +, setindex!, filter!

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
mutable struct ParameterBackpropagateContext{T <: Trace} <: BackpropagationContext
    tr::T
    weight::Float64
    visited::Visitor
    params::ParameterStore
    param_grads::Gradients
end
ParameterBackpropagate(tr::T, params) where T <: Trace = ParameterBackpropagateContext(tr, 0.0, Visitor(), params, Gradients())
ParameterBackpropagate(tr::T, params, param_grads::Gradients) where {T <: Trace, K <: UnconstrainedSelection} = ParameterBackpropagateContext(tr, 0.0, Visitor(), params, param_grads)

# Choice sites
mutable struct ChoiceBackpropagateContext{T <: Trace, K <: UnconstrainedSelection} <: BackpropagationContext
    tr::T
    weight::Float64
    visited::Visitor
    params::ParameterStore
    choice_grads::Gradients
    select::K
end
ChoiceBackpropagate(tr::T, params, choice_grads) where {T <: Trace, K <: UnconstrainedSelection} = ChoiceBackpropagateContext(tr, 0.0, Visitor(), params, choice_grads, UnconstrainedAllSelection())
ChoiceBackpropagate(tr::T, params, choice_grads, sel::K) where {T <: Trace, K <: UnconstrainedSelection} = ChoiceBackpropagateContext(tr, 0.0, Visitor(), params, choice_grads, sel)

# ------------ Choice sites ------------ #

@inline function (ctx::ParameterBackpropagateContext)(call::typeof(rand), 
                                                      addr::T, 
                                                      d::Distribution{K}) where {T <: Address, K}
    s = get_choice(ctx.tr, addr).val
    ctx.weight += logpdf(d, s)
    #visit!(ctx.visited, addr)
    return s
end

@inline function (ctx::ChoiceBackpropagateContext)(call::typeof(rand), 
                                                   addr::T, 
                                                   d::Distribution{K}) where {T <: Address, K}
    s = get_choice(ctx.tr, addr).val
    ctx.weight += logpdf(d, s)
    #visit!(ctx.visited, addr)
    return s
end

# ------------ Learnable ------------ #

read_parameter(ctx::K, addr::Address) where K <: BackpropagationContext = read_parameter(ctx, ctx.params, addr)
read_parameter(ctx::K, params::ParameterStore, addr::Address) where K <: BackpropagationContext = ctx.tr.params[addr].val

Zygote.@adjoint function read_parameter(ctx, params, addr)
    ret = read_parameter(ctx, params, addr)
    fn = param_grad -> begin
        state_grad = nothing
        params_grad = ParameterStore(Dict{Address, Any}(addr => param_grad))
        (state_grad, params_grad, nothing)
    end
    return ret, fn
end

@inline function (ctx::ParameterBackpropagateContext)(fn::typeof(learnable), addr::Address, p::T) where T
    return read_parameter(ctx, addr)
end

@inline function (ctx::ChoiceBackpropagateContext)(fn::typeof(learnable), addr::Address, p::T) where T
    return read_parameter(ctx, addr)
end

# ------------ Call sites ------------ #

# Learnable parameters.
simulate_call_pullback(param_grads, cl::T, args) where T <: CallSite = cl.ret

Zygote.@adjoint function simulate_call_pullback(param_grads, cl, args)
    ret = simulate_call_pullback(param_grads, cl, args)
    fn = ret_grad -> begin
        arg_grads = accumulate_parameter_gradients!(param_grads, cl, ret_grad)
        (nothing, nothing, arg_grads)
    end
    return ret, fn
end

@inline function (ctx::ParameterBackpropagateContext)(c::typeof(rand),
                                                      addr::T,
                                                      call::Function,
                                                      args...) where T <: Address
    #visit!(ctx.visited, addr)
    cl = get_call(ctx.tr, addr)
    param_grads = Gradients()
    ret = simulate_call_pullback(param_grads, cl, args)
    ctx.param_grads.tree[addr] = param_grads
    return ret
end

# Choices.
simulate_choice_pullback(choice_grads, choice_selection, cl::T, args) where T <: CallSite = cl.ret

Zygote.@adjoint function simulate_choice_pullback(choice_grads, choice_selection, cl, args)
    ret = simulate_choice_pullback(choice_grads, choice_selection, cl, args)
    fn = ret_grad -> begin
        arg_grads, choice_vals, choice_grads = choice_gradients(choice_grads, choice_selection, cl, ret_grad)
        (nothing, nothing, (choice_vals, choice_grads), arg_grads)
    end
    return ret, fn
end

@inline function (ctx::ChoiceBackpropagateContext)(c::typeof(rand),
                                                   addr::T,
                                                   call::Function,
                                                   args...) where T <: Address
    #visit!(ctx.visited, addr)
    cl = get_call(ctx.tr, addr)
    choice_grads = Gradients()
    ret = simulate_choice_pullback(choice_grads, get_sub(ctx.select, addr), cl, args)
    ctx.choice_grads.tree[addr] = choice_grads
    return ret
end

# ------------ Accumulate gradients ------------ #

function accumulate_parameter_gradients!(param_grads, cl::T, ret_grad, scaler::Float64 = 1.0) where T <: CallSite
    fn = (args, params) -> begin
        ctx = ParameterBackpropagate(cl.trace, params, param_grads)
        ret = ctx(cl.fn, args...)
        (ctx.weight, ret)
    end
    blank = ParameterStore()
    _, back = Zygote.pullback(fn, cl.args, blank)
    arg_grads, ps_grad = back((1.0, ret_grad))
    if !(ps_grad isa Nothing)
        for (addr, grad) in ps_grad.params
            push!(param_grads, addr, scaler * grad)
        end
    end
    return arg_grads
end

# ------------ Compute choice gradients ------------ #

function filter!(choice_grads, cl::BlackBoxCallSite, grad_tr::NamedTuple, sel::K) where K <: UnconstrainedSelection
    values = ConstrainedHierarchicalSelection()
    for (k, v) in cl.trace.choices
        has_query(sel, k) && begin
            push!(values, k, v.val)
            push!(choice_grads, k, grad_tr.choices[k].val)
        end
    end
    return values
end

function choice_gradients(choice_grads, choice_selection, cl, ret_grad)
    call = cl.fn
    fn = (args, trace) -> begin
        ctx = ChoiceBackpropagate(trace, ParameterStore(), choice_grads, choice_selection)
        ret = ctx(call, args...)
        (ctx.weight, ret)
    end
    _, back = Zygote.pullback(fn, cl.args, cl.trace)
    arg_grads, grad_ref = back((1.0, ret_grad))
    gs_trace = grad_ref[]
    choice_vals = filter!(choice_grads, cl, gs_trace, choice_selection)
    return arg_grads, choice_vals, choice_grads
end

# ------------ Convenience getters ------------ #

function get_choice_gradients(cl::T, ret_grad) where T <: CallSite
    choice_grads = Gradients()
    choice_selection = UnconstrainedAllSelection()
    choice_gradients(choice_grads, choice_selection, cl, ret_grad)
    return choice_grads
end

function get_parameter_gradients(cl::T, ret_grad, scaler::Float64 = 1.0) where T <: CallSite
    param_grads = Gradients()
    accumulate_parameter_gradients!(param_grads, cl, ret_grad, scaler)
    return param_grads
end

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct ParameterBackpropagateContext{T <: Trace} <: BackpropagationContext
    tr::T
    weight::Float64
    visited::Visitor
    params::ParameterStore
    param_grads::Gradients
end
```
`ParameterBackpropagateContext` is used to compute the gradients of parameters with respect to following objective:

Outer constructors:
```julia
ParameterBackpropagate(tr::T, params) where T <: Trace = ParameterBackpropagateContext(tr, 0.0, Visitor(), params, Gradients())
ParameterBackpropagate(tr::T, params, param_grads::Gradients) where {T <: Trace, K <: UnconstrainedSelection} = ParameterBackpropagateContext(tr, 0.0, Visitor(), params, param_grads)
```
""", ParameterBackpropagateContext)

@doc(
"""
```julia
mutable struct ChoiceBackpropagateContext{T <: Trace} <: BackpropagationContext
    tr::T
    weight::Float64
    visited::Visitor
    params::ParameterStore
    param_grads::Gradients
end
```
`ChoiceBackpropagateContext` is used to compute the gradients of choices with respect to following objective:

Outer constructors:
```julia
ParameterBackpropagate(tr::T, params) where T <: Trace = ChoiceBackpropagateContext(tr, 0.0, Visitor(), params, Gradients())
ChoiceBackpropagate(tr::T, params, param_grads::Gradients) where {T <: Trace, K <: UnconstrainedSelection} = ChoiceBackpropagateContext(tr, 0.0, Visitor(), params, param_grads)
```
""", ChoiceBackpropagateContext)

@doc(
"""
```julia
gradients = get_choice_gradients(cl::T, ret_grad) where T <: CallSite
````

Returns a `Gradients` object which tracks the gradients with respect to the objective of random choices with differentiable `logpdf` in the program.
""", get_choice_gradients)
