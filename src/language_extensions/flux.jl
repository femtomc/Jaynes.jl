(ctx::Jaynes.SimulateContext)(fn::typeof(deep), model, args...) = model(args...)
(ctx::Jaynes.GenerateContext)(fn::typeof(deep), model, args...) = model(args...)
(ctx::Jaynes.UpdateContext)(fn::typeof(deep), model, args...) = model(args...)
(ctx::Jaynes.RegenerateContext)(fn::typeof(deep), model, args...) = model(args...)
(ctx::Jaynes.ProposeContext)(fn::typeof(deep), model, args...) = model(args...)
(ctx::Jaynes.ScoreContext)(fn::typeof(deep), model, args...) = model(args...)

struct ModelParameterStore
    params::IdDict
    ModelParameterStore() = new(IdDict())
    ModelParameterStore(p::IdDict) = new(p)
end
haskey(ps::ModelParameterStore, model) = haskey(ps.params, model)
setindex!(ps::ModelParameterStore, val, model) = ps.params[model] = val

Zygote.@adjoint ModelParameterStore(params) = ModelParameterStore(params), store_grad -> (nothing,)

function +(a::ModelParameterStore, b::ModelParameterStore)
    params = IdDict()
    for (k, v) in Iterators.flatten((a.params, b.params))
        if !haskey(params, k)
            params[k] = v
        else
            params[k] += v
        end
    end
    ModelParameterStore(params)
end

mutable struct FluxNetworkTrainContext{T <: Jaynes.CallSite, 
                                       S <: Jaynes.AddressMap, 
                                       P <: Jaynes.AddressMap} <: Jaynes.BackpropagationContext
    call::T
    weight::Float64
    scaler::Float64
    fixed::S
    network_params::ModelParameterStore
    initial_params::P
end

function DeepBackpropagate(fixed, params, call::T, model_grads, scaler) where T <: Jaynes.CallSite
    FluxNetworkTrainContext(call,
                            0.0,
                            scaler,
                            fixed,
                            model_grads,
                            params)
end

apply_model(ctx, params, model, args...) = model(args...)

Zygote.@adjoint function apply_model(ctx, params, model, args...)
    ret = model(args...)
    fn = params_grad -> begin
        _, back = Zygote.pullback((m, x) -> m(x...), model, args)
        gs, arg_grads = back(params_grad)
        scaled_grads = -ctx.scaler * Flux.destructure(gs)[1]
        (nothing, ModelParameterStore(IdDict(model => scaled_grads)), nothing, arg_grads...)
    end
    return ret, fn
end

simulate_deep_pullback(fixed, params, cl::T, args) where T <: Jaynes.CallSite = get_ret(cl)

Zygote.@adjoint function simulate_deep_pullback(fixed, params, cl::T, args) where T <: Jaynes.CallSite
    ret = simulate_deep_pullback(fixed, params, cl, args)
    fn = ret_grad -> begin
        arg_grads = accumulate_deep_gradients(fixed, params, cl, ret_grad)
        (nothing, nothing, nothing, arg_grads)
    end
    ret, fn
end

function accumulate_deep_gradients(fx, ps, cl, ret_grad, scaler)
    fn = (args, model_grads) -> begin
        ctx = DeepBackpropagate(fx, ps, cl, model_grads, scaler)
        ret = ctx(cl.fn, args...)
        (ctx.weight, ret)
    end
    blank = ModelParameterStore()
    _, back = Zygote.pullback(fn, cl.args, blank)
    arg_grads, model_grads = back((1.0, ret_grad))
    arg_grads, model_grads
end

function (ctx::FluxNetworkTrainContext)(fn::typeof(Jaynes.deep), 
                                        model,
                                        args...) where A <: Jaynes.Address
    apply_model(ctx, ctx.network_params, model, args...)
end

@inline function (ctx::FluxNetworkTrainContext)(call::typeof(rand), 
                                                addr::T, 
                                                d::Distribution{K}) where {T <: Jaynes.Address, K}
    if haskey(ctx.fixed, addr)
        s = getindex(ctx.fixed, addr)
    else
        s = Jaynes.get_value(Jaynes.get_sub(ctx.call, addr))
    end
    Jaynes.increment!(ctx, logpdf(d, s))
    return s
end

@inline function (ctx::FluxNetworkTrainContext)(c::typeof(rand),
                                                addr::T,
                                                call::Function,
                                                args...) where T <: Jaynes.Address
    cl = get_sub(ctx.call, addr)
    fx = get_sub(ctx.fixed, addr)
    ps = get_sub(ctx.initial_params, addr)
    ret = simulate_deep_pullback(fx, ps, cl, args)
    return ret
end

function get_deep_gradients(ps::P, cl::C, ret_grad; scaler = 1.0) where {P <: Jaynes.AddressMap, C <: Jaynes.CallSite}
    arg_grads, model_grads = accumulate_deep_gradients(Jaynes.Empty(), ps, cl, ret_grad, scaler)
    return arg_grads, model_grads
end

function get_deep_gradients(cl::C, ret_grad; scaler = 1.0) where {P <: Jaynes.AddressMap, C <: Jaynes.CallSite}
    arg_grads, model_grads = accumulate_deep_gradients(Jaynes.Empty(), Jaynes.Empty(), cl, ret_grad, scaler)
    return arg_grads, model_grads
end

function one_shot_neural_gradient_estimator(tg::K,
                                                 ps::P,
                                                 v_mod::Function,
                                                 v_args::Tuple,
                                                 mod::Function,
                                                 args::Tuple;
                                                 scale = 1.0) where {K <: Jaynes.AddressMap, P <: Jaynes.AddressMap}
    _, cl = simulate(ps, v_mod, v_args...)
    merge!(cl, tg) && error("(one_shot_neural_gradient_estimator): variational model proposes to addresses in observations.")
    _, mlw = score(cl, ps, mod, args...)
    lw = mlw - get_score(cl)
    _, model_grads = get_deep_gradients(ps, cl, 1.0; scaler = lw * scale)
    return model_grads, lw, cl
end

function one_shot_neural_gradient_estimator(tg::K,
                                            v_mod::Function,
                                            v_args::Tuple,
                                            mod::Function,
                                            args::Tuple;
                                            scale = 1.0) where K <: Jaynes.AddressMap
    one_shot_neural_gradient_estimator(tg,
                                       Jaynes.Empty(),
                                       v_mod,
                                       v_args,
                                       mod,
                                       args;
                                       scale = scale)
end

function accumulate!(d1::IdDict, d2::ModelParameterStore)
    for (m, gs) in d2.params
        if haskey(d1, m)
            d1[m] += gs
        else
            d1[m] = gs
        end
    end
end

function update_models!(opt, d1::IdDict)
    for (m, gs) in d1
        ps, re = Flux.destructure(m)
        update!(opt, ps, gs)
        new = Flux.params(re(ps))
        Flux.loadparams!(m, new)
    end
end

function neural_variational_inference!(opt,
                                       tg::K,
                                       ps::P,
                                       v_mod::Function,
                                       v_args::Tuple,
                                       mod::Function,
                                       args::Tuple;
                                       gs_samples = 100) where {K <: Jaynes.AddressMap, P <: Jaynes.AddressMap}
    elbo_est = 0.0
    gs_est = IdDict()
    lws = Vector{Float64}(undef, gs_samples)
    for s in 1 : gs_samples
        model_grads, lws[i], cl = osng(tg, ps, 
                                       v_mod, v_args, 
                                       mod, args; 
                                       scale = 1.0 / gs_samples)
        elbo_est += lws[i] / gs_samples
        accumulate!(gs_est, model_grads)
    end
    update_models!(opt, gs_est)
    cl = cs[rand(Categorical(nw(lws)))]
    elbo_est, cl
end

function neural_variational_inference!(opt,
                                       tg::K,
                                       v_mod::Function,
                                       v_args::Tuple,
                                       mod::Function,
                                       args::Tuple;
                                       gs_samples = 100) where {K <: Jaynes.AddressMap, P <: Jaynes.AddressMap}
    neural_variational_inference!(opt,
                                  tg, 
                                  Jaynes.Empty(), 
                                  v_mod, 
                                  v_args, 
                                  mod, 
                                  args; 
                                  gs_samples = gs_samples)
end

function vimco_neural_gradient_estimator(tg::K,
                                         ps::P,
                                         v_mod::Function,
                                         v_args::Tuple,
                                         mod::Function,
                                         args::Tuple;
                                         est_samples::Int = 100,
                                         scale = 1.0) where {K <: Jaynes.AddressMap, P <: Jaynes.AddressMap}
    cs = Vector{Jaynes.CallSite}(undef, est_samples)
    lws = Vector{Float64}(undef, est_samples)
    Threads.@threads for i in 1:est_samples
        _, cs[i] = simulate(ps, v_mod, v_args...)
        merge!(cs[i], tg) && error("(vimco_gradient_estimator): variational model proposes to addresses in observations.")
        ret, mlw = score(cs[i], ps, mod, args...)
        lws[i] = mlw - get_score(cs[i])
    end
    ltw = Jaynes.lse(lws)
    L = ltw - log(est_samples)
    nw = exp.(lws .- ltw)
    gs_est = IdDict()
    bs = Jaynes.geometric_base(lws)
    Threads.@threads for i in 1 : est_samples
        ls = L - nw[i] - bs[i]
        accumulate!(gs_est, get_deep_gradients(ps, cs[i], 1.0; scaler = ls * scale)[2])
    end
    return gs_est, L, cs, nw
end

function neural_geometric_vimco!(opt,
                                 tg::K,
                                 ps::P,
                                 est_samples::Int,
                                 v_mod::Function,
                                 v_args::Tuple,
                                 mod::Function,
                                 args::Tuple;
                                 gs_samples = 100) where {K <: Jaynes.AddressMap, P <: Jaynes.AddressMap}
    velbo_est = 0.0
    gs_est = IdDict()
    cls = Vector{Jaynes.CallSite}(undef, gs_samples)
    for s in 1 : gs_samples
        gs, L, cs, nw = vimges(tg, ps, 
                               v_mod, v_args, 
                               mod, args; 
                               est_samples = est_samples, scale = 1.0 / gs_samples)
        velbo_est += L / gs_samples
        accumulate!(gs_est, gs)
        cls[s] = cs[rand(Categorical(nw))]
    end
    update_models!(opt, gs_est)
    velbo_est, cls
end

function neural_geometric_vimco!(opt,
                                 tg::K,
                                 est_samples::Int,
                                 v_mod::Function,
                                 v_args::Tuple,
                                 mod::Function,
                                 args::Tuple;
                                 gs_samples = 100) where {K <: Jaynes.AddressMap, P <: Jaynes.AddressMap}
    neural_geometric_vimco!(opt,
                            tg, 
                            Jaynes.Empty(), 
                            est_samples, 
                            v_mod, 
                            v_args, 
                            mod, 
                            args; 
                            gs_samples = gs_samples)
end
