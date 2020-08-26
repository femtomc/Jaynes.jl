macro load_flux_fmi()
    @info "Loading differentiable compatibility to \u001b[3m\u001b[34;1mFlux.jl\u001b[0m\n\n          \u001b[34;1mhttps://github.com/FluxML/Flux.jl\n "

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
            fixed::S
            initial_params::P
            opt
        end

        function DeepBackpropagate(call::T, fixed, params) where T <: Jaynes.CallSite
            FluxNetworkTrainContext(call,
                                    0.0,
                                    fixed,
                                    params,
                                    ADAM())
        end

        apply_model!(ctx, model, args...) = model(args...)

        Zygote.@adjoint function apply_model!(ctx, model, args...)
            ret = model(args...)
            fn = params_grad -> begin
                _, back = Zygote.pullback((m, x) -> m(x...), model, args)
                gs, arg_grads = back(params_grad)
                _, re = Flux.destructure(Flux.params(model))
                ps = Flux.destructure(model)[1]
                update!(ctx.opt, ps, Flux.destructure(gs)[1])
                new = Flux.params(Flux._restructure(model, ps))
                Flux.loadparams!(model, new)
                (nothing, nothing, arg_grads...)
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

        function accumulate_deep_gradients!(fx, ps, cl, ret_grad)
            fn = args -> begin
                ctx = DeepBackpropagate(cl, fx, ps)
                ret = ctx(cl.fn, args...)
                (ctx.weight, ret)
            end
            _, back = Zygote.pullback(fn, cl.args)
            arg_grads = back((1.0, ret_grad))[1]
            arg_grads
        end

        function (ctx::FluxNetworkTrainContext)(fn::typeof(Jaynes.deep), 
                                                model,
                                                args...) where A <: Jaynes.Address
            ret = apply_model!(ctx, model, args...)
            ret
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

        function deep_train!(ps::P, cl::C, ret_grad) where {P <: Jaynes.AddressMap, C <: Jaynes.CallSite}
            arg_grads = accumulate_deep_gradients!(Jaynes.Empty(), ps, cl, ret_grad)
            return arg_grads
        end

        function deep_train!(cl::C, ret_grad) where {P <: Jaynes.AddressMap, C <: Jaynes.CallSite}
            arg_grads = accumulate_deep_gradients!(Jaynes.Empty(), Jaynes.Empty(), cl, ret_grad)
            return arg_grads
        end
    end

    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
