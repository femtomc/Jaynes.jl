macro load_turing_fmi()
    expr = quote
        @info "Loading foreign model interface to \u001b[3m\u001b[34;1mTuring.jl\u001b[0m\n\n          \u001b[34;1mhttps://turing.ml/dev/\n\n "

        import Jaynes: has_top, get_top, has_sub, get_sub, get_score, collect!
        using Turing
        using Turing.Inference: tilde, dot_tilde
        
        z# ------------ Trace ------------ #

        mutable struct TuringTrace{NT <: NamedTuple} <: Jaynes.Trace
            choices::NT
            TuringTrace(nt::NT) where NT = new{NT}(nt)
            TuringTrace{NT}() where NT = new{NT}()
        end
        add_call!(st::TuringTrace, addr, choices) = st.choices = choices

        # ------------ Call site ------------ #

        mutable struct TuringModelCallSite{M, A} <: Jaynes.CallSite
            trace::TuringTrace
            score::Float64
            model::M
            args::A
        end

        get_score(gfcs::TuringModelCallSite) = gfcs.score
        haskey(cs::TuringModelCallSite, addr) = haskey(trace, addr)
        getindex(cs::TuringModelCallSite, addrs...) = getindex(trace, addr)
        get_ret(cs::TuringModelCallSite) = cs.trace

        # Overwrite Turing methods.
        function (ctx::Jaynes.SimulateContext)(c::typeof(foreign),
                                               addr::A,
                                               model::M) where {A <: Jaynes.Address, M <: Turing.Model}
            Jaynes.visit!(ctx, addr)
            mf = model.mf
            tr, w = Jaynes.simulate(mf, ctx, ctx, ctx, model)
            ret = Jaynes.get_retval(tr)
            Jaynes.add_call!(ctx, addr, TuringModelCallSite(TuringTrace(tr), Turing.get_score(tr), model, args, ret))
            Jaynes.increment!(ctx, w)
            return ret
        end

        function (ctx::Jaynes.GenerateContext)(c::typeof(foreign),
                                               addr::A,
                                               model::M) where {A <: Jaynes.Address, M <: Turing.Model}
            Jaynes.visit!(ctx, addr)
            sel = get_sub(ctx.select, addr)
            mf = model.mf
            tr, w = Jaynes.generate(sel, mf, ctx, ctx, ctx, model)
            ret = Jaynes.get_retval(tr)
            Jaynes.add_call!(ctx, addr, TuringModelCallSite(TuringTrace(tr), Turing.get_score(tr), model, args, ret))
            Jaynes.increment!(ctx, w)
            return ret
        end
    end

    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
