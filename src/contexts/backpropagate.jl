import Base: +, setindex!

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

mutable struct BackpropagateContext{T <: Trace} <: ExecutionContext
    tr::T
    score::Float64
    visited::Visitor
    params::ParameterStore
    param_grads::ParameterGradients
end
Backpropagate(tr::T, params) where T <: Trace = BackpropagateContext(tr, 0.0, Visitor(), params, ParameterGradients())

# ------------ Choice sites ------------ #

@inline function (ctx::BackpropagateContext)(call::typeof(rand), 
                                             addr::T, 
                                             d::Distribution{K}) where {T <: Address, K}
    s = get_choice(ctx.tr, addr).val
    ctx.score += logpdf(d, s)
    #visit!(ctx.visited, addr)
    return s
end

# ------------ Learnable ------------ #

read_parameter(ctx::BackpropagateContext, addr::Address) = read_parameter(ctx, ctx.params, addr)
read_parameter(ctx::BackpropagateContext, params::ParameterStore, addr::Address) = ctx.tr.params[addr].val

Zygote.@adjoint function read_parameter(ctx, params, addr)
    ret = read_parameter(ctx, params, addr)
    fn = param_grad -> begin
        state_grad = nothing
        params_grad = ParameterStore(Dict{Address, Any}(addr => param_grad))
        (state_grad, params_grad, nothing)
    end
    return ret, fn
end

@inline function (ctx::BackpropagateContext)(fn::typeof(learnable), addr::Address, p::T) where T
    return read_parameter(ctx, addr)
end

# ------------ Call sites ------------ #

simulate_call_pullback(param_grads, cl::T, args) where {T <: CallSite, K <: UnconstrainedSelection} = cl.ret

Zygote.@adjoint function simulate_call_pullback(param_grads, cl, args)
    ret = simulate_call_pullback(param_grads, cl, args)
    fn = ret_grad -> begin
        arg_grads = accumulate_param_gradients!(param_grads, cl, ret_grad)
        (nothing, nothing, arg_grads)
    end
    (ret, fn)
end

@inline function (ctx::BackpropagateContext)(c::typeof(rand),
                                             addr::T,
                                             call::Function,
                                             args...) where T <: Address
    #visit!(ctx.visited, addr)
    cl = get_call(ctx.tr, addr)
    rg_ctx = Backpropagate(cl.trace, ParameterStore())
    ret = simulate_call_pullback(rg_ctx.param_grads, cl, args)
    ctx.param_grads.tree[addr] = rg_ctx.param_grads
    return ret
end

function accumulate_param_gradients!(param_grads, cl::T, ret_grad) where T <: CallSite
    fn = (args, params) -> begin
        ctx = Backpropagate(cl.trace, params)
        ret = ctx(cl.fn, args...)
        (ctx.score, ret)
    end
    blank = ParameterStore()
    _, back = Zygote.pullback(fn, cl.args, blank)
    arg_grads, ps_grad = back((1.0, ret_grad))
    if !(ps_grad isa Nothing)
        for (addr, grad) in ps_grad.params
            push!(param_grads, addr, grad)
        end
    end
    return arg_grads
end

function parameter_gradients(cl::T, ret_grad) where T <: CallSite
    param_grads = ParameterGradients()
    accumulate_param_gradients!(param_grads, cl, ret_grad)
    return param_grads
end
