macro load_gen_fmi()
    @info "Loading foreign model interface to \u001b[3m\u001b[34;1mGen.jl\u001b[0m\n\n          \u001b[34;1mhttps://www.gen.dev/\n\nThis interface currently supports Gen's full feature set.\n\n\u001b[1mGen and Jaynes share exports - please qualify usage of the following context APIs:\n\u001b[0m\n \u001b[31msimulate   \u001b[0m-> \u001b[32mJaynes.simulate\n \u001b[31mgenerate   \u001b[0m-> \u001b[32mJaynes.generate\n \u001b[31mupdate     \u001b[0m-> \u001b[32mJaynes.update\n \u001b[31mregenerate \u001b[0m-> \u001b[32mJaynes.regenerate\n \u001b[31mpropose    \u001b[0m-> \u001b[32mJaynes.propose\n "
    expr = quote

        import Jaynes: simulate, propose, generate, update, regenerate, score
        
        using Gen
        import Gen: get_choices

        # ------------ Address map ------------ #
        
        struct GenerativeFunctionLeaf{T <: Gen.Trace} <: Leaf{GenerativeFunctionLeaf}
            tr::T
        end
        
        @inline convert(::Type{Value}, c::GenerativeFunctionLeaf) = Value(get_value(c.tr))
        
        const GenerativeFunctionTrace = Jaynes.SoloMap{GenerativeFunctionLeaf}
        GenerativeFunctionTrace(gen_tr::T) where T <: Gen.Trace = GenerativeFunctionTrace(GenerativeFunctionLeaf(gen_tr))

        # ------------ Call site ------------ #

        struct GenerativeFunctionCallSite{M <: Gen.GenerativeFunction, A, K} <: Jaynes.CallSite
            map::GenerativeFunctionTrace
            score::Float64
            model::M
            args::A
            ret::K
        end
        @inline get_model(gfcs::GenerativeFunctionCallSite) = gfcs.model

        @inline isempty(gfcs::GenerativeFunctionCallSite) = false

        function projection(tr::GenerativeFunctionTrace, tg::Target)
            weight = 0.0
            for (k, v) in shallow_iterator(tr)
                ss = get_sub(tg, k)
                weight += projection(v, ss)
            end
            weight
        end

        @inline projection(gfcs::GenerativeFunctionCallSite, tg::Jaynes.Empty) = 0.0
        @inline projection(gfcs::GenerativeFunctionCallSite, tg::Jaynes.SelectAll) = get_score(gfcs)
        @inline projection(gfcs::GenerativeFunctionCallSite, tg::Target) = project(get_trace(gfcs), tg)

        @inline filter(fn, gfcs::GenerativeFunctionCallSite) = filter(fn, get_trace(gfcs))
        @inline filter(fn, addr, gfcs::GenerativeFunctionCallSite) = filter(fn, addr, get_trace(gfcs))

        @inline select(gfcs::GenerativeFunctionCallSite) = select(get_trace(gfcs))

        # ------------ Contexts ------------ #

        function (ctx::Jaynes.SimulateContext)(c::typeof(foreign),
                                               addr::A,
                                               gen_fn::M,
                                               args...) where {A <: Jaynes.Address, M <: Gen.GenerativeFunction}
            Jaynes.visit!(ctx, addr)
            ret, cl = simulate(gen_fn, args)
            Jaynes.add_call!(ctx, addr, cl)
            return ret
        end

        # Convenience.
        function simulate(gen_fn::G, args...) where G <: Gen.GenerativeFunction
            ctx = Jaynes.Simulate()
            ret = ctx(foreign, gen_fn, args...)
            cl = GenerativeFunctionCallSite(GenerativeFunctionTrace(ctx.tr), Gen.get_score(ctx.tr), gen_fn, args, Gen.get_retval(tr))
            return ret, cl
        end

        function (ctx::Jaynes.ProposeContext)(c::typeof(foreign),
                                              addr::A,
                                              gen_fn::M,
                                              args...) where {A <: Jaynes.Address, M <: Gen.GenerativeFunction}
            Jaynes.visit!(ctx, addr)
            ret, cl, w = propose(gen_fn, args, choice_map)
            Jaynes.add_call!(ctx, addr, cl)
            Jaynes.increment!(ctx, w)
            return ret
        end

        # Convenience.
        function propose(gen_fn::G, args...) where G <: Gen.GenerativeFunction
            ctx = Propose()
            ret = ctx(foreign, gen_fn, args...)
            cl = GenerativeFunctionCallSite(GenerativeFunctionTrace(ctx.tr), Gen.get_score(ctx.tr), gen_fn, args, Gen.get_retval(tr))
            return ret, cl, ctx.score
        end

        function (ctx::Jaynes.GenerateContext)(c::typeof(foreign),
                                               addr::A,
                                               gen_fn::M,
                                               args...) where {A <: Jaynes.Address, M <: Gen.GenerativeFunction}
            Jaynes.visit!(ctx, addr)
            constraints = Jaynes.dump_queries(Jaynes.get_sub(ctx.select, addr))
            pairs = create_pairs(constraints)
            choice_map = Gen.choicemap(pairs...)
            tr, w = Gen.generate(gen_fn, args, choice_map)
            ret = Gen.get_retval(tr)
            Jaynes.add_call!(ctx, addr, GenerativeFunctionCallSite(GenTrace(tr), Gen.get_score(tr), gen_fn, args, ret))
            Jaynes.increment!(ctx, w)
            return ret
        end

        # Convenience.
        function generate(sel::L, gen_fn::G, args...) where {L <: Jaynes.AddressMap, G <: Gen.GenerativeFunction}
            addr = gensym()
            v_sel = selection(addr => sel)
            ctx = Generate(v_sel)
            ret = ctx(foreign, addr, gen_fn, args...)
            return ret, get_sub(ctx.tr, addr), ctx.weight
        end

        function (ctx::Jaynes.UpdateContext)(c::typeof(foreign),
                                             addr::A,
                                             gen_fn::M,
                                             args...) where {A <: Jaynes.Address, M <: Gen.GenerativeFunction}
            Jaynes.visit!(ctx, addr)
            constraints = Jaynes.dump_queries(Jaynes.get_sub(ctx.select, addr))
            pairs = create_pairs(constraints)
            choice_map = Gen.choicemap(pairs...)
            prev = Jaynes.get_prev(ctx, addr)
            new, w, rd, d = Gen.update(get_gen_trace(prev), args, (), choice_map)
            Jaynes.add_call!(ctx, addr, GenerativeFunctionCallSite(GenTrace(new), Gen.get_score(new), gen_fn, args, Gen.get_retval(new)))
            Jaynes.increment!(ctx, w)
            return Gen.get_retval(new)
        end

        function (ctx::Jaynes.UpdateContext{C})(c::typeof(foreign),
                                                gen_fn::M,
                                                args...) where {A <: Jaynes.Address, M <: Gen.GenerativeFunction, C <: GenerativeFunctionCallSite}
            constraints = Jaynes.dump_queries(ctx.select)
            pairs = create_pairs(constraints)
            choice_map = Gen.choicemap(pairs...)
            prev = ctx.prev
            new, w, rd, d = Gen.update(get_gen_trace(prev), args, (), choice_map)
            ctx.tr.tr = new
            ctx.score += Gen.get_score(new)
            Jaynes.increment!(ctx, w)
            return Gen.get_retval(new)
        end

        # Convenience.
        function update(sel::L, gen_cl::C) where {L <: Jaynes.AddressMap, C <: GenerativeFunctionCallSite}
            ctx = Jaynes.UpdateContext(gen_cl, sel, Jaynes.NoChange())
            ret = ctx(foreign, gen_cl.model, gen_cl.args...)
            return ret, GenerativeFunctionCallSite(ctx.tr, ctx.score, gen_cl.model, gen_cl.args, ret), ctx.weight, Jaynes.UndefinedChange(), nothing
        end

        function (ctx::Jaynes.RegenerateContext)(c::typeof(foreign),
                                                 addr::A,
                                                 gen_fn::M,
                                                 args...) where {A <: Jaynes.Address, M <: Gen.GenerativeFunction}
            Jaynes.visit!(ctx, addr)
            constraints = Jaynes.dump_queries(Jaynes.get_sub(ctx.select, addr))
            select = Gen.select(constraints...)
            prev = Jaynes.get_prev(ctx, addr)
            new, w, rd = Gen.regenerate(prev.trace.tr, args, (), select)
            ret = Gen.get_retval(new)
            Jaynes.add_call!(ctx, addr, GenerativeFunctionCallSite(GenTrace(new), Gen.get_score(new), gen_fn, args, ret))
            Jaynes.increment!(ctx, Gen.get_score(new) - Jaynes.get_score(prev))
            return ret
        end

        function (ctx::Jaynes.RegenerateContext{C})(c::typeof(foreign),
                                                    gen_fn::M,
                                                    args...) where {A <: Jaynes.Address, M <: Gen.GenerativeFunction, C <: GenerativeFunctionCallSite}
            constraints = Jaynes.dump_queries(ctx.select)
            select = Gen.select(constraints...)
            prev = ctx.prev
            new, w, rd = Gen.regenerate(get_gen_trace(prev), args, (), select)
            ret = Gen.get_retval(new)
            ctx.tr.tr = new
            ctx.score += Gen.get_score(new)
            Jaynes.increment!(ctx, Gen.get_score(new) - Jaynes.get_score(prev))
            return ret
        end

        # Convenience.
        function regenerate(sel::L, gen_cl::C) where {L <: Jaynes.Target, C <: GenerativeFunctionCallSite}
            ctx = Jaynes.Regenerate(gen_cl, sel, Jaynes.NoChange())
            ret = ctx(foreign, gen_cl.model, gen_cl.args...)
            return ret, GenerativeFunctionCallSite(ctx.tr, ctx.score, gen_cl.model, gen_cl.args, ret), ctx.weight, Jaynes.UndefinedChange(), nothing
        end

        function (ctx::Jaynes.ScoreContext)(c::typeof(foreign),
                                            addr::Jaynes.Address,
                                            gen_fn::M,
                                            args...) where M <: Gen.GenerativeFunction
            Jaynes.visit!(ctx, addr)
            constraints = Jaynes.dump_queries(Jaynes.get_sub(ctx.select, addr))
            pairs = create_pairs(constraints)
            choice_map = Gen.choicemap(pairs...)
            w, ret = Gen.assess(gen_fn, args, choice_map)
            Jaynes.increment!(ctx, w)
            return ret
        end

        # Convenience.
        function score(sel::L, gen_fn::M, args...) where {L <: Jaynes.Target, M <: GenerativeFunctionCallSite}
            addr = gensym()
            v_sel = selection(addr => sel)
            ctx = Jaynes.Score(v_sel)
            ret = ctx(foreign, addr, gen_fn, args...)
            return ret, ctx.weight
        end

        # ------------ Pretty printing ------------ #

        function collect!(par::P, addrs::Vector{Any}, chd::Dict{Any, Any}, chm::M, meta) where {P <: Tuple, M <: Gen.ChoiceMap}
            for (k, v) in get_values_shallow(chm)
                push!(addrs, (par..., k))
                chd[(par..., k)] = v
                meta[(par..., k)] = "(Gen)"
            end
            for (k, v) in get_submaps_shallow(chm)
                collect!((par..., k), addrs, chd, v, meta)
            end
        end

        function collect!(par::P, addrs::Vector{Any}, chd::Dict{Any, Any}, tr::T, meta) where {P <: Tuple, T <: GenTrace}
            choices = get_choices(tr.tr)
            for (k, v) in get_values_shallow(choices)
                push!(addrs, (par..., k))
                chd[(par..., k)] = v
                meta[(par..., k)] = "(Gen)"
            end
            for (k, v) in get_submaps_shallow(choices)
                collect!((par..., k), addrs, chd, v, meta)
            end
        end

        function collect!(addrs::Vector{Any}, chd::Dict{Any, Any}, chm::M, meta) where M <: Gen.ChoiceMap
            for (k, v) in get_values_shallow(chm)
                push!(addrs, (k, ))
                chd[(k, )] = v
            end
            for (k, v) in get_submaps_shallow(chm)
                collect!((k, ), addrs, chd, v, meta)
            end
        end

        function collect!(addrs::Vector{Any}, chd::Dict{Any, Any}, tr::T, meta) where T <: GenTrace
            choices = get_choices(tr.tr)
            for (k, v) in get_values_shallow(choices)
                push!(addrs, (k, ))
                chd[(k, )] = v
            end
            for (k, v) in get_submaps_shallow(choices)
                collect!((k, ), addrs, chd, v, meta)
            end
        end

        # ------------ Exchange kernel ------------ #

        apply_kernel(ker, cl::GenerativeFunctionCallSite) = ker(get_gen_trace(cl))

    end

    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
