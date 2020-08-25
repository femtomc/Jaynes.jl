macro load_flux_fmi()
    @info "Loading differentiable compatibility to \u001b[3m\u001b[34;1mFlux.jl\u001b[0m\n\n          \u001b[34;1mhttps://github.com/FluxML/Flux.jl\n "

    expr = quote

        using Flux
        using Flux: Chain, Dense, update!

        function (ctx::Jaynes.SimulateContext)(fn::typeof(deep), 
                                               addr::A, 
                                               model, 
                                               args...) where A <: Jaynes.Address
            model(args...)
        end

        function (ctx::Jaynes.GenerateContext)(fn::typeof(deep), 
                                               addr::A, 
                                               model, 
                                               args...) where A <: Jaynes.Address
            model(args...)
        end

        function (ctx::Jaynes.UpdateContext)(fn::typeof(deep), 
                                             addr::A, 
                                             model, 
                                             args...) where A <: Jaynes.Address
            model(args...)
        end

        function (ctx::Jaynes.RegenerateContext)(fn::typeof(deep), 
                                                 addr::A, 
                                                 model, 
                                                 args...) where A <: Jaynes.Address
            model(args...)
        end

        function (ctx::Jaynes.ProposeContext)(fn::typeof(deep), 
                                              addr::A, 
                                              model, 
                                              args...) where A <: Jaynes.Address
            model(args...)
        end

        function (ctx::Jaynes.ScoreContext)(fn::typeof(deep), 
                                            addr::A, 
                                            model, 
                                            args...) where A <: Jaynes.Address
            model(args...)
        end

        mutable struct FluxNetworkBackpropagateContext{T <: Jaynes.CallSite, 
                                                       S <: Jaynes.AddressMap, 
                                                       P <: Jaynes.AddressMap} <: Jaynes.BackpropagationContext
            call::T
            weight::Float64
            fixed::S
            initial_params::P
            net_params::Jaynes.ParameterStore
            net_params_grads::Jaynes.Gradients
        end

        function DeepBackpropagate(call::T, fixed, params, net_params) where T <: Jaynes.CallSite
            FluxNetworkBackpropagateContext(call,
                                            0.0,
                                            fixed,
                                            params,
                                            net_params,
                                            Jaynes.Gradients())
        end
        
        function DeepBackpropagate(call::T, fixed, params, net_params, grads) where T <: Jaynes.CallSite
            FluxNetworkBackpropagateContext(call,
                                            0.0,
                                            fixed,
                                            params,
                                            net_params,
                                            grads)
        end
        
        apply_model(ctx, addr, model, args...) = model(args...)

        Zygote.@adjoint function apply_model(ctx, addr, model, args...)
            ret = model(args...)
            fn = params_grad -> begin
                _, back = Zygote.pullback((m, x) -> m(x...), model, args)
                gs, arg_grads = back(params_grad)
                Jaynes.set_sub!(ctx.net_params_grads, addr, Jaynes.Value(gs))
                (nothing, nothing, nothing, arg_grads...)
            end
            return ret, fn
        end

        simulate_deep_pullback(fixed, params, grads, cl::T, args) where T <: Jaynes.CallSite = get_ret(cl)

        Zygote.@adjoint function simulate_deep_pullback(fixed, params, grads, cl::T, args) where T <: Jaynes.CallSite
            ret = simulate_deep_pullback(fixed, params, grads, cl, args)
            fn = ret_grad -> begin
                arg_grads = accumulate_deep_gradients!(fixed, params, grads, cl, ret_grad)
                (nothing, nothing, nothing, nothing, arg_grads)
            end
            ret, fn
        end

        function accumulate_deep_gradients!(fx, ps, net_grads, cl, ret_grad)
            fn = (args, net_params) -> begin
                ctx = DeepBackpropagate(cl, fx, ps, net_params, net_grads)
                ret = ctx(cl.fn, args...)
                (ctx.weight, ret)
            end
            blank = Jaynes.ParameterStore()
            _, back = Zygote.pullback(fn, cl.args, blank)
            arg_grads, acc_net_grads = back((1.0, ret_grad))
            if !(acc_net_grads isa Nothing)
                for (addr, grad) in acc_net_grads.params
                    accumulate!(net_grads, addr, grad)
                end
            end
            arg_grads
        end

        function (ctx::FluxNetworkBackpropagateContext)(fn::typeof(Jaynes.deep), 
                                                        addr::A, 
                                                        model, 
                                                        args...) where A <: Jaynes.Address
            ret = apply_model(ctx, addr, model, args...)
            ret
        end

        @inline function (ctx::FluxNetworkBackpropagateContext)(call::typeof(rand), 
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

        @inline function (ctx::FluxNetworkBackpropagateContext)(c::typeof(rand),
                                                                addr::T,
                                                                call::Function,
                                                                args...) where T <: Jaynes.Address
            cl = get_sub(ctx.call, addr)
            fx = get_sub(ctx.fixed, addr)
            ps = get_sub(ctx.initial_params, addr)
            net_grads = Gradients()
            ret = simulate_deep_pullback(fx, ps, net_grads, cl, args)
            set_sub!(ctx.net_params_grads, addr, net_grads)
            return ret
        end

        function get_deep_gradients(ps::P, cl::C, ret_grad) where {P <: Jaynes.AddressMap, C <: Jaynes.CallSite}
            net_grads = Jaynes.Gradients()
            arg_grads = accumulate_deep_gradients!(target(), ps, net_grads, cl, ret_grad)
            return arg_grads, net_grads
        end

        function get_deep_gradients(cl::C, ret_grad) where {P <: Jaynes.AddressMap, C <: Jaynes.CallSite}
            net_grads = Jaynes.Gradients()
            arg_grads = accumulate_deep_gradients!(target(), Jaynes.Empty(), net_grads, cl, ret_grad)
            return arg_grads, net_grads
        end
    end

    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
