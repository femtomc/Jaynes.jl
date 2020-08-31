macro load_flux_fmi()
    @info "Loading differentiable compatibility interface to \u001b[3m\u001b[34;1mFlux.jl\u001b[0m\n\n          \u001b[34;1mhttps://github.com/FluxML/Flux.jl\n "

    expr = quote

        using Flux
        using Flux: Chain, Dense, update!

        (ctx::Jaynes.SimulateContext)(fn::typeof(deep), model, args...) = model(args...)
        (ctx::Jaynes.GenerateContext)(fn::typeof(deep), model, args...) = model(args...)
        (ctx::Jaynes.UpdateContext)(fn::typeof(deep), model, args...) = model(args...)

        (ctx::Jaynes.RegenerateContext)(fn::typeof(deep), model, args...) = model(args...)
        (ctx::Jaynes.ProposeContext)(fn::typeof(deep), model, args...) = model(args...)
        (ctx::Jaynes.ScoreContext)(fn::typeof(deep), model, args...) = model(args...)

        mutable struct FluxNetworkTrainContext{T <: Jaynes.CallSite, 
                                               S <: Jaynes.AddressMap, 
                                               P <: Jaynes.AddressMap} <: Jaynes.BackpropagationContext
            call::T
            weight::Float64
            scaler::Float64
            fixed::S
            network_params::IdDict
            learnables::P
            opt
        end

        function DeepBackpropagate(fixed, params, call::T, model_grads, opt, scaler) where T <: Jaynes.CallSite
            FluxNetworkTrainContext(call,
                                    0.0,
                                    scaler,
                                    fixed,
                                    model_grads,
                                    params,
                                    opt)
        end

        apply_model!(ctx, params, model, args...) = model(args...)

        Zygote.@adjoint function apply_model!(ctx, params, model, args...)
            ret = model(args...)
            fn = params_grad -> begin
                _, back = Zygote.pullback((m, x) -> m(x...), model, args)
                gs, arg_grads = back(params_grad)
                scaled_grads = -ctx.scaler * Flux.destructure(gs)[1]
                (nothing, IdDict(model => scaled_grads), nothing, arg_grads...)
            end
            return ret, fn
        end

        simulate_deep_pullback(fixed, params, cl::T, args) where T <: Jaynes.CallSite = get_ret(cl)

        Zygote.@adjoint function simulate_deep_pullback(fixed, params, cl::T, args) where T <: Jaynes.CallSite
            ret = simulate_deep_pullback(fixed, params, cl, args)
            fn = ret_grad -> begin
                arg_grads = accumulate_deep_gradients!(fixed, params, cl, ret_grad)
                (nothing, nothing, nothing, arg_grads)
            end
            ret, fn
        end

        function accumulate_deep_gradients!(fx, ps, cl, ret_grad, opt, scaler)
            fn = (args, model_grads) -> begin
                ctx = DeepBackpropagate(fx, ps, cl, model_grads, opt, scaler)
                ret = ctx(cl.fn, args...)
                (ctx.weight, ret)
            end
            blank = IdDict()
            _, back = Zygote.pullback(fn, cl.args, blank)
            arg_grads, model_grads = back((1.0, ret_grad))
            arg_grads, model_grads
        end

        function (ctx::FluxNetworkTrainContext)(fn::typeof(Jaynes.deep), 
                                                model,
                                                args...) where A <: Jaynes.Address
            apply_model!(ctx, ctx.network_params, model, args...)
        end

        @inline function (ctx::FluxNetworkTrainContext)(fn::typeof(learnable), addr::Jaynes.Address)
            haskey(ctx.learnables, addr) && return getindex(ctx.learnables, addr)
            error("Parameter not provided at address $addr.")
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
            ps = get_sub(ctx.learnables, addr)
            ret = simulate_deep_pullback(fx, ps, cl, args)
            return ret
        end

        function get_deep_gradients!(ps::P, cl::C, ret_grad; opt = ADAM(), scaler = 1.0) where {P <: Jaynes.AddressMap, C <: Jaynes.CallSite}
            arg_grads, model_grads = accumulate_deep_gradients!(Jaynes.Empty(), ps, cl, ret_grad, opt, scaler)
            return arg_grads, model_grads
        end

        function get_deep_gradients!(cl::C, ret_grad; opt = ADAM(), scaler = 1.0) where {P <: Jaynes.AddressMap, C <: Jaynes.CallSite}
            arg_grads, model_grads = accumulate_deep_gradients!(Jaynes.Empty(), Jaynes.Empty(), cl, ret_grad, opt, scaler)
            return arg_grads, model_grads
        end

        function one_shot_neural_gradient_estimator_step!(tg::K,
                                                          ps::P,
                                                          v_mod::Function,
                                                          v_args::Tuple,
                                                          mod::Function,
                                                          args::Tuple;
                                                          opt = ADAM(),
                                                          scale = 1.0) where {K <: Jaynes.AddressMap, P <: Jaynes.AddressMap}
            _, cl = simulate(ps, v_mod, v_args...)
            merge!(cl, tg) && error("(one_shot_neural_gradient_estimator_step!): variational model proposes to addresses in observations.")
            _, mlw = score(cl, ps, mod, args...)
            lw = mlw - get_score(cl)
            _, model_grads = get_deep_gradients!(ps, cl, 1.0; opt = opt, scaler = lw * scale)
            return model_grads, lw, cl
        end

        function one_shot_neural_gradient_estimator_step!(tg::K,
                                                          v_mod::Function,
                                                          v_args::Tuple,
                                                          mod::Function,
                                                          args::Tuple;
                                                          opt = ADAM(),
                                                          scale = 1.0) where K <: Jaynes.AddressMap
            one_shot_neural_gradient_estimator_step!(tg,
                                                     Jaynes.Empty(),
                                                     v_mod,
                                                     v_args,
                                                     mod,
                                                     args;
                                                     opt = opt,
                                                     scale = scale)
        end

        const osnges! = one_shot_neural_gradient_estimator_step!

        function accumulate!(d1::IdDict, d2::IdDict)
            for (m, gs) in d2
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

        function neural_variational_inference!(tg::K,
                                               ps::P,
                                               v_mod::Function,
                                               v_args::Tuple,
                                               mod::Function,
                                               args::Tuple;
                                               opt = ADAM(0.05, (0.9, 0.8)),
                                               n_iters = 1000,
                                               gs_samples = 100) where {K <: Jaynes.AddressMap, P <: Jaynes.AddressMap}
            cls = Vector{Jaynes.CallSite}(undef, n_iters)
            elbows = Vector{Float64}(undef, n_iters)
            Threads.@threads for i in 1 : n_iters
                elbo_est = 0.0
                gs_est = IdDict()
                for s in 1 : gs_samples
                    model_grads, lw, cl = osnges!(tg, ps, 
                                                  v_mod, v_args, 
                                                  mod, args; 
                                                  opt = opt, scale = 1.0 / gs_samples)
                    elbo_est += lw / gs_samples
                    accumulate!(gs_est, model_grads)
                    cls[i] = cl
                end
                elbows[i] = elbo_est
                @info "ELBO estimate: $elbo_est"
                update_models!(opt, gs_est)
            end
            elbows, cls
        end

        function neural_variational_inference!(tg::K,
                                               v_mod::Function,
                                               v_args::Tuple,
                                               mod::Function,
                                               args::Tuple;
                                               opt = ADAM(0.05, (0.9, 0.8)),
                                               n_iters = 1000,
                                               gs_samples = 100) where {K <: Jaynes.AddressMap, P <: Jaynes.AddressMap}
            neural_variational_inference!(tg, 
                                          Jaynes.Empty(), 
                                          v_mod, 
                                          v_args, 
                                          mod, 
                                          args; 
                                          opt = opt, 
                                          n_iters = n_iters,
                                          gs_samples = gs_samples)
        end

        const nvi! = neural_variational_inference!

        function vimco_neural_gradient_estimator_step!(tg::K,
                                                       ps::P,
                                                       v_mod::Function,
                                                       v_args::Tuple,
                                                       mod::Function,
                                                       args::Tuple;
                                                       opt = ADAM(),
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
                accumulate!(gs_est, get_deep_gradients!(ps, cs[i], 1.0; opt = opt, scaler = ls * scale)[2])
            end
            return gs_est, L, cs, nw
        end

        const vimges! = vimco_neural_gradient_estimator_step!

        function neural_geometric_vimco!(tg::K,
                                         ps::P,
                                         est_samples::Int,
                                         v_mod::Function,
                                         v_args::Tuple,
                                         mod::Function,
                                         args::Tuple;
                                         opt = ADAM(0.05, (0.9, 0.8)),
                                         n_iters = 1000,
                                         gs_samples = 100) where {K <: Jaynes.AddressMap, P <: Jaynes.AddressMap}
            cls = Vector{Jaynes.CallSite}(undef, n_iters)
            velbows = Vector{Float64}(undef, n_iters)
            Threads.@threads for i in 1 : n_iters
                velbo_est = 0.0
                gs_est = IdDict()
                for s in 1 : gs_samples
                    gs, L, cs, nw = vimges!(tg, ps, 
                                            v_mod, v_args, 
                                            mod, args; 
                                            opt = opt, est_samples = est_samples, scale = 1.0 / gs_samples)
                    velbo_est += L / gs_samples
                    accumulate!(gs_est, gs)
                    cls[s] = cs[rand(Categorical(nw))]
                end
                velbows[i] = velbo_est
                update_models!(opt, gs_est)
            end
            velbows, cls
        end

        function neural_geometric_vimco!(tg::K,
                                         est_samples::Int,
                                         v_mod::Function,
                                         v_args::Tuple,
                                         mod::Function,
                                         args::Tuple;
                                         opt = ADAM(0.05, (0.9, 0.8)),
                                         n_iters = 1000,
                                         gs_samples = 100) where {K <: Jaynes.AddressMap, P <: Jaynes.AddressMap}
            neural_geometric_vimco!(tg, 
                                    Jaynes.Empty(), 
                                    est_samples, 
                                    v_mod, 
                                    v_args, 
                                    mod, 
                                    args; 
                                    opt = opt, 
                                    n_iters = n_iters,
                                    gs_samples = gs_samples)
        end

        const nvimco! = neural_geometric_vimco!
    end

    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
