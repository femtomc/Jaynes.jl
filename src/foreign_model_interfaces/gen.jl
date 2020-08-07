macro load_gen_fmi()
    @info "Loading foreign model interface to \u001b[3m\u001b[34;1mGen.jl\u001b[0m\n\n          \u001b[34;1mhttps://www.gen.dev/\n\nThis interface currently supports Gen's full feature set.\n\n\u001b[1mGen and Jaynes share exports - please qualify usage of the following context APIs:\n\u001b[0m\n \u001b[31msimulate   \u001b[0m-> \u001b[32mJaynes.simulate\n \u001b[31mgenerate   \u001b[0m-> \u001b[32mJaynes.generate\n \u001b[31mupdate     \u001b[0m-> \u001b[32mJaynes.update\n \u001b[31mregenerate \u001b[0m-> \u001b[32mJaynes.regenerate\n \u001b[31mpropose    \u001b[0m-> \u001b[32mJaynes.propose\n "
    expr = quote
        import Jaynes: has_top, get_top, has_sub, get_sub, get_score, collect, collect!, selection, get_selection, get_ret
        import Jaynes: simulate, propose, generate, regenerate, update, score
        import Jaynes: apply_kernel
        import Base.getindex
        
        using Gen
        import Gen: get_choices

        # ------------ Trace ------------ #

        mutable struct GenTrace{T <: Gen.Trace} <: Jaynes.Trace
            tr::T
            GenTrace(tr::T) where T = new{T}(tr)
            GenTrace{T}() where T = new{T}()
        end
        has_top(cs::GenTrace, addr) where T <: Jaynes.Address = has_value(Gen.get_choices(cs), addr)
        has_top(cs::GenTrace, addr::Tuple{T}) where T <: Jaynes.Address = has_top(cs, addr[1])
        get_top(cs::GenTrace, addr) where T <: Jaynes.Address = get_value(Gen.get_choices(cs), addr)
        get_top(cs::GenTrace, addr::Tuple{T}) where T <: Jaynes.Address = get_value(Gen.get_choices(cs), addr[1])
        has_sub(cs::GenTrace, addr) = has_value(Gen.get_choices(cs), addr)
        has_sub(cs::GenTrace, addr::Tuple{T}) where T <: Jaynes.Address = has_sub(cs, addr[1])
        get_sub(cs::GenTrace, addr) where T <: Jaynes.Address = get_value(Gen.get_choices(cs), addr)
        get_sub(cs::GenTrace, addr::Tuple{T}) where T <: Jaynes.Address = get_sub(cs, addr[1])
        get_choices(tr::GenTrace) = Gen.get_choices(tr.tr)
        
        # ------------ Call site ------------ #

        struct GenerativeFunctionCallSite{M <: Gen.GenerativeFunction, A, K} <: Jaynes.CallSite
            trace::GenTrace
            score::Float64
            model::M
            args::A
            ret::K
        end

        get_score(gfcs::GenerativeFunctionCallSite) = gfcs.score
        get_gen_trace(cs::GenerativeFunctionCallSite) = cs.trace.tr
        get_ret(cs::GenerativeFunctionCallSite) = get_retval(get_gen_trace(cs))
        get_choices(cs::GenerativeFunctionCallSite) = get_choices(get_gen_trace(cs))
        has_top(cs::GenerativeFunctionCallSite, addr) = has_top(cs.trace, addr)
        get_top(cs::GenerativeFunctionCallSite, addr) = get_top(cs.trace, addr)
        has_sub(cs::GenerativeFunctionCallSite, addr) = has_sub(cs.trace, addr)
        get_sub(cs::GenerativeFunctionCallSite, addr) = get_sub(cs.trace, addr)
        haskey(cs::GenerativeFunctionCallSite, addr) = has_value(get_choices(cs.trace), addr)
        getindex(cs::GenerativeFunctionCallSite, addr) = getindex(get_gen_trace(cs), addr)

        # ------------ Choice map integration ------------ #

        function create_pairs(v::Vector{Pair})
            out = []
            for (t, l) in v
                push!(out, (foldr(=>, t), l))
            end
            out
        end

        function selection(chm::C) where C <: Gen.ChoiceMap
            s = Jaynes.ConstrainedHierarchicalSelection()
            for (k, v) in Gen.get_values_shallow(chm)
                push!(s, k, v)
            end
            for (k, v) in Gen.get_submaps_shallow(chm)
                sub = selection(v)
                s.tree[k] = sub
            end
            s
        end

        get_selection(tr::T) where T <: Gen.Trace = selection(Gen.get_choices(tr))

        # ------------ Contexts ------------ #

        function (ctx::Jaynes.SimulateContext)(c::typeof(foreign),
                                               addr::A,
                                               gen_fn::M,
                                               args...) where {A <: Jaynes.Address, M <: Gen.GenerativeFunction}
            Jaynes.visit!(ctx, addr)
            tr = Gen.simulate(gen_fn, args)
            Jaynes.add_call!(ctx, addr, GenerativeFunctionCallSite(GenTrace(tr), Gen.get_score(tr), gen_fn, args, Gen.get_retval(tr)))
            return Gen.get_retval(tr)
        end

        # Convenience.
        function simulate(gen_fn::G, args...) where G <: Gen.GenerativeFunction
            ctx = Jaynes.Simulate()
            addr = gensym()
            ret = ctx(foreign, addr, gen_fn, args...)
            return ret, get_sub(ctx.tr, addr)
        end

        function (ctx::Jaynes.ProposeContext)(c::typeof(foreign),
                                              addr::A,
                                              gen_fn::M,
                                              args...) where {A <: Jaynes.Address, M <: Gen.GenerativeFunction}
            Jaynes.visit!(ctx, addr)
            tr, w, ret = Gen.propose(gen_fn, args, choice_map)
            Jaynes.add_call!(ctx, addr, GenerativeFunctionCallSite(GenTrace(tr), Gen.get_score(tr), gen_fn, args, Gen.get_retval(tr)))
            Jaynes.increment!(ctx, w)
            return ret
        end

        # Convenience.
        function propose(gen_fn::G, args...) where G <: Gen.GenerativeFunction
            ctx = Propose()
            addr = gensym()
            ret = ctx(foreign, addr, gen_fn, args...)
            return ret, get_top(ctx.tr, addr), ctx.score
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
        function generate(sel::L, gen_fn::G, args...) where {L <: Jaynes.ConstrainedSelection, G <: Gen.GenerativeFunction}
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
        function update(sel::L, gen_cl::C) where {L <: Jaynes.ConstrainedSelection, C <: GenerativeFunctionCallSite}
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
        function regenerate(sel::L, gen_cl::C) where {L <: Jaynes.UnconstrainedSelection, C <: GenerativeFunctionCallSite}
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
        function score(sel::L, gen_fn::M, args...) where {L <: Jaynes.UnconstrainedSelection, M <: GenerativeFunctionCallSite}
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
