macro load_gen()
    expr = quote
        using Gen

        struct GenCallSite{T, M <: GenerativeFunction, K} <: Jaynes.CallSite
            trace::T
            score::Float64
            model::M
            args::Tuple
            ret::K
        end

        function (ctx::Jaynes.SimulateContext)(c::typeof(gen_fmi),
                                               fn::typeof(rand),
                                               addr::Address,
                                               call::M,
                                               args...) where M <: GenerativeFunction
            tr = Gen.generate(call, args, choice_map)
            Jaynes.add_call!(ctx, addr, GenCallSite(tr, get_score(tr), call, args, get_retval(tr)))
        end

        function (ctx::Jaynes.GenerateContext)(c::typeof(gen_fmi),
                                               fn::typeof(rand),
                                               addr::Address,
                                               call::M,
                                               args...) where M <: GenerativeFunction
            choice_map = Jaynes.get_sub(ctx.select, addr)
            tr, w = Gen.generate(call, args, choice_map)
            Jaynes.add_call!(ctx, addr, Jaynes.GenCallSite(tr, get_score(tr), call, args, get_retval(tr)))
            Jaynes.increment!(ctx, w)
        end

        function (ctx::Jaynes.UpdateContext)(c::typeof(gen_fmi),
                                      fn::typeof(rand),
                                      addr::Address,
                                      call::M,
                                      args...) where M <: GenerativeFunction
            choice_map = Jaynes.get_sub(ctx.select, addr)
            prev = Jaynes.get_prev(ctx, addr)
            new, w, rd, d = Gen.generate(prev.trace, args, (), choice_map)
            Jaynes.add_call!(ctx, addr, Jaynes.GenCallSite(tr, get_score(tr), call, args, get_retval(tr)))
            Jaynes.increment!(ctx, w)
        end
    end

    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
