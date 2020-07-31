macro load_gen_fmi()
    @info "Loading foreign model interface to \u001b[4m\u001b[32mGen.jl\u001b[0m\n\nThis interface currently supports Gen's full feature set.\n\n\u001b[1mGen and Jaynes share exports - please qualify usage of the following context APIs:\n\u001b[0m\n \u001b[31msimulate   \u001b[0m-> \u001b[32mJaynes.simulate\n \u001b[31mgenerate   \u001b[0m-> \u001b[32mJaynes.generate\n \u001b[31mupdate     \u001b[0m-> \u001b[32mJaynes.update\n \u001b[31mregenerate \u001b[0m-> \u001b[32mJaynes.regenerate\n "
    expr = quote
        import Jaynes: has_top, get_top, has_sub, get_sub, get_score, collect!
        using Gen

        # ------------ Call site ------------ #

        struct GenerativeFunctionCallSite{T <: Gen.Trace, M <: GenerativeFunction, A, K} <: Jaynes.CallSite
            trace::T
            score::Float64
            model::M
            args::A
            ret::K
        end

        get_score(gfcs::GenerativeFunctionCallSite) = gfcs.score
        haskey(cs::GenerativeFunctionCallSite, addr) = has_value(get_choices(cs.trace), addr)
        getindex(cs::GenerativeFunctionCallSite, addrs...) = getindex(get_choices(cs.trace), addrs...)
        get_ret(cs::GenerativeFunctionCallSite) = get_retval(cs.trace)

        # ------------ Pretty printing ------------ #

        function collect!(par::P, addrs::Vector{Any}, chd::Dict{Any, Any}, tr::T, meta) where {P <: Tuple, T <: Gen.Trace}
            choices = get_choices(tr)
            for (k, v) in get_values_shallow(choices)
                push!(addrs, (par..., k))
                chd[(par..., k)] = v
                meta[(par..., k)] = "(Gen)"
            end
            for (k, v) in get_submaps_shallow(choices)
                collect!((par..., k), addrs, chd, v.trace, meta)
            end
        end

        function collect!(addrs::Vector{Any}, chd::Dict{Any, Any}, tr::T, meta) where T <: Gen.Trace
            choices = get_choices(tr)
            for (k, v) in get_values_shallow(choices)
                push!(addrs, (k, ))
                chd[(k, )] = v
            end
            for (k, v) in get_submaps_shallow(choices)
                collect!((k, ), addrs, chd, v.trace, meta)
            end
        end

        # ------------ Contexts ------------ #

        function (ctx::Jaynes.SimulateContext)(c::typeof(gen_fmi),
                                               addr::Jaynes.Address,
                                               call::M,
                                               args...) where M <: GenerativeFunction
            tr = Gen.simulate(call, args)
            Jaynes.add_call!(ctx, addr, GenerativeFunctionCallSite(tr, Gen.get_score(tr), call, args, Gen.get_retval(tr)))
        end

        function (ctx::Jaynes.GenerateContext)(c::typeof(gen_fmi),
                                               addr::Jaynes.Address,
                                               call::M,
                                               args...) where M <: GenerativeFunction
            choice_map = Jaynes.get_sub(ctx.select, addr)
            tr, w = Gen.generate(call, args, choice_map)
            Jaynes.add_call!(ctx, addr, Jaynes.GenerativeFunctionCallSite(tr, Gen.get_score(tr), call, args, Gen.get_retval(tr)))
            Jaynes.increment!(ctx, w)
        end

        function (ctx::Jaynes.UpdateContext)(c::typeof(gen_fmi),
                                             addr::Jaynes.Address,
                                             call::M,
                                             args...) where M <: GenerativeFunction
            choice_map = Jaynes.get_sub(ctx.select, addr)
            prev = Jaynes.get_prev(ctx, addr)
            new, w, rd, d = Gen.generate(prev.trace, args, (), choice_map)
            Jaynes.add_call!(ctx, addr, Jaynes.GenerativeFunctionCallSite(tr, Gen.get_score(tr), call, args, Gen.get_retval(tr)))
            Jaynes.increment!(ctx, w)
        end
    end

    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
