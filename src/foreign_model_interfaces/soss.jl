macro load_soss_fmi()
    expr = quote
        @info "Loading foreign model interface to \u001b[3m\u001b[34;1mSoss.jl\u001b[0m\n\n          \u001b[34;1mhttps://github.com/cscherrer/Soss.jl\n\nThis interface is a work in progress! Please post issues to:\n\n          \u001b[34;1mhttps://github.com/femtomc/Jaynes.jl/issues\n\n"

        using PyCall
        using Conda

        # Install SymPy.
        if PyCall.conda
            Conda.add("pip")
            pip = joinpath(Conda.BINDIR, "pip")
            run(`$pip install sympy`)
        else
            try
                pyimport("sympy")
            catch ee
                typeof(ee) <: PyCall.PyError || rethrow(ee)
                warn("""
                     Python Dependencies not installed
                     Please either:
                     - Rebuild PyCall to use Conda, by running in the julia REPL:
                     - `ENV[PYTHON]=""; Pkg.build("PyCall"); Pkg.build("Jaynes")`
                     - Or install the dependencies: `pip install sympy`
                     """)
            end
        end

        import Jaynes: has_top, get_top, has_sub, get_sub, get_score, collect!, add_call!, haskey
        import Jaynes: simulate, propose, generate, update, regenerate, score
        using Soss
        import Soss: logpdf

        # ------------ Trace ------------ #

        mutable struct SossTrace{NT <: NamedTuple} <: Jaynes.Trace
            choices::NT
            SossTrace(nt::NT) where NT = new{NT}(nt)
            SossTrace{NT}() where NT = new{NT}()
        end
        add_call!(st::SossTrace, addr, choices) = st.choices = choices

        # ------------ Call site ------------ #

        mutable struct SossModelCallSite{M, A} <: Jaynes.CallSite
            trace::SossTrace
            score::Float64
            model::M
            args::A
        end

        get_score(gfcs::SossModelCallSite) = gfcs.score
        haskey(cs::SossModelCallSite, addr) = haskey(trace, addr)
        getindex(cs::SossModelCallSite, addrs...) = getindex(trace, addr)
        get_ret(cs::SossModelCallSite) = cs.trace

        # ------------ Pretty printing ------------ #

        function collect!(par::P, addrs::Vector{Any}, chd::Dict{Any, Any}, tr::ST, meta) where {P <: Tuple, ST <: SossTrace}
            for (k, v) in pairs(tr.choices)
                push!(addrs, (par..., k))
                chd[(par..., k)] = v
                meta[(par..., k)] = "(Soss)"
            end
        end

        function collect!(addrs::Vector{Any}, chd::Dict{Any, Any}, tr::ST, meta) where ST <: SossTrace
            for (k, v) in pairs(tr.choices)
                push!(addrs, (k, ))
                chd[(k, )] = v
            end
        end

        # ------------ Contexts ------------ #

        function (ctx::Jaynes.SimulateContext)(c::typeof(foreign),
                                               addr::T,
                                               model::M,
                                               args...) where {T <: Jaynes.Address, M <: Soss.Model}
            choices = rand(model(args...))
            score = Soss.logpdf(m(args...), choices)
            Jaynes.add_call!(ctx, addr, SossModelCallSite(SossTrace(choices), score, model, args))
            return choices
        end
        
        # Convenience.
        function simulate(model::M, args...) where M <: Soss.Model
            ctx = Jaynes.Simulate()
            addr = gensym()
            ret = ctx(foreign, addr, model, args...)
            return ret, get_sub(ctx.tr, addr)
        end

        function (ctx::Jaynes.ProposeContext)(c::typeof(foreign),
                                              addr::T,
                                              model::M,
                                              args...) where {T <: Jaynes.Address, M <: Soss.Model}
            choices = rand(model(args...))
            score = Soss.logpdf(m(args...), choices)
            Jaynes.add_call!(ctx, addr, SossModelCallSite(SossTrace(choices), score, model, args))
            increment!(ctx, score)
            return choices
        end
       
        # Convenience.
        function propose(model::M, args...) where M <: Soss.Model
            ctx = Propose()
            addr = gensym()
            ret = ctx(foreign, addr, model, args...)
            return ret, get_top(ctx.tr, addr), ctx.score
        end

        function (ctx::Jaynes.GenerateContext)(c::typeof(foreign),
                                               addr::T,
                                               model::M,
                                               args...) where {T <: Jaynes.Address, M <: Soss.Model}
            data = Jaynes.get_top(ctx.select, addr)
            w, choices = Soss.weightedSample(model(args...), data)
            score = Soss.logpdf(m(args...), choices)
            Jaynes.add_call!(ctx, addr, SossModelCallSite(SossTrace(choices), score, model, args))
            Jaynes.increment!(ctx, w)
            return choices
        end

        # Convenience.
        function generate(sel::L, model::M, args...) where {L <: NamedTuple, M <: Soss.Model}
            addr = gensym()
            v_sel = selection([(addr, ) => sel])
            ctx = Generate(v_sel)
            ret = ctx(foreign, addr, model, args...)
            return ret, get_sub(ctx.tr, addr), ctx.weight
        end

        function (ctx::Jaynes.UpdateContext)(c::typeof(foreign),
                                             addr::T,
                                             model::M,
                                             args...) where {T <: Jaynes.Address, M <: Soss.Model}
            prev = Jaynes.get_prev(ctx, addr)
            kvs = Jaynes.get_top(ctx.select, addr)
            data = namedtuple(Dict{Symbol, Any}(kvs))
            w, choices = Soss.weightedSample(model(args...), data)
            score = Soss.logpdf(m(args...), choices)
            Jaynes.add_call!(ctx, addr, SossModelCallSite(SossTrace(choices), score, model, args))
            Jaynes.increment!(ctx, w - get_score(prev))
            return choices
        end
        
        # Convenience.
        function update(sel::L, soss_cl::C) where {L <: NamedTuple, C <: SossModelCallSite}
            addr = gensym()
            v_sel = selection([(addr, ) => sel])
            ctx = Update(v_sel)
            ret = ctx(foreign, addr, model, args...)
            return ret, get_sub(ctx.tr, addr), ctx.weight, Jaynes.UndefinedChange(), nothing
        end

        function (ctx::Jaynes.RegenerateContext{C, T})(c::typeof(foreign),
                                                       addr::A,
                                                       model::M,
                                                       args...) where {A <: Jaynes.Address, M <: Soss.Model, C <: SossModelCallSite, T <: SossTrace}
            targeted = dump_queries(Jaynes.get_sub(ctx.select, addr))
            prev = ctx.prev
            kvs = Dict{Symbol, Any}()
            for (k, v) in pairs(prev.trace.choices)
                haskey(ctx.select, (addr, k)) && continue
                kvs[k] = v
            end
            data = Soss.namedtuple(kvs)
            w, choices = Soss.weightedSample(model(args...), data)
            score = Soss.logpdf(m(args...), choices)
            ctx.tr.choices = choices
            ctx.score += score
            Jaynes.increment!(ctx, score - get_score(prev))
            return choices
        end

        function (ctx::Jaynes.RegenerateContext)(c::typeof(foreign),
                                                 addr::T,
                                                 model::M,
                                                 args...) where {T <: Jaynes.Address, M <: Soss.Model}
            targeted = dump_queries(Jaynes.get_sub(ctx.select, addr))
            prev = Jaynes.get_prev(ctx, addr)
            kvs = Dict{Symbol, Any}()
            for (k, v) in pairs(prev.trace.choices)
                k in targeted && continue
                kvs[k] = v
            end
            data = Soss.namedtuple(kvs)
            w, choices = Soss.weightedSample(model(args...), data)
            score = Soss.logpdf(m(args...), choices)
            Jaynes.add_call!(ctx, addr, SossModelCallSite(SossTrace(choices), score, model, args))
            Jaynes.increment!(ctx, score - get_score(prev))
            return choices
        end

        # Convenience.
        function regenerate(sel::L, soss_cl::C) where {L <: Jaynes.UnconstrainedSelection, C <: SossModelCallSite}
            addr = gensym()
            v_sel = selection(addr => sel)
            ctx = Jaynes.Regenerate(soss_cl, v_sel, NoChange())
            ret = ctx(foreign, addr, soss_cl.model, soss_cl.args...)
            return ret, SossModelCallSite(ctx.tr, ctx.score, soss_cl.model, soss_cl.args), ctx.weight, Jaynes.UndefinedChange(), nothing
        end

        function (ctx::Jaynes.ScoreContext)(c::typeof(foreign),
                                            addr::T,
                                            model::M,
                                            args...) where {T <: Jaynes.Address, M <: Soss.Model}
            kvs = Jaynes.get_sub(ctx.select, addr)
            choices = namedtuple(Dict{Symbol, Any}(kvs))
            score = Soss.logpdf(m(args...), choices)
            Jaynes.increment!(ctx, score)
            return choices
        end

        # Convenience.
        function score(sel::L, model::M, args...) where {L <: Jaynes.UnconstrainedSelection, M <: SossModelCallSite}
            addr = gensym()
            v_sel = selection(addr => sel)
            ctx = Jaynes.Score(v_sel)
            ret = ctx(foreign, addr, soss_cl.model, soss_cl.args...)
            return ret, SossModelCallSite(ctx.tr, ctx.score, soss_cl.model, soss_cl.args), ctx.weight, Jaynes.UndefinedChange(), nothing
        end
    end

    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
