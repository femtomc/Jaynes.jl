macro load_gen_fmi()
    @info "Loading foreign model interface to Gen.jl.\nThis interface currently supports Gen's full feature set."
    expr = quote
        import Jaynes: has_top, get_top, has_sub, get_sub, get_score, collect!
        using Gen

        # ------------ Call site ------------ #

        struct GenerativeFunctionCallSite{T, M <: GenerativeFunction, K} <: Jaynes.CallSite
            trace::T
            score::Float64
            model::M
            args::Tuple
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
