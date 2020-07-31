macro load_soss_fmi()
    expr = quote
        @info "Loading foreign model interface to \u001b[3m\u001b[34;1mSoss.jl\u001b[0m\n\n          \u001b[34;1mhttps://github.com/cscherrer/Soss.jl\n\n "

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
                     """
                    )
            end
        end

        using Soss

        # ------------ Call site ------------ #

        mutable struct SossModelCallSite{NT, M, F} <: Jaynes.CallSite
            trace::NT
            score::Float64
            model::M
            args::Tuple
        end

        get_score(gfcs::SossModelCallSite) = gfcs.score
        haskey(cs::SossModelCallSite, addr) = haskey(trace, addr)
        getindex(cs::SossModelCallSite, addrs...) = getindex(trace, addr)
        get_ret(cs::SossModelCallSite) = cs.trace

        # ------------ Contexts ------------ #

        function (ctx::Jaynes.SimulateContext)(c::typeof(soss_fmi),
                                               addr::T,
                                               model::M,
                                               args...) where {T <: Jaynes.Address, M <: Soss.Model}
            choices = rand(model(args...))
            score = Soss.logpdf(args, choices)
            Jaynes.add_call!(ctx, addr, SossModelCallSite(choices, score, model, args))
            return choices
        end

        function (ctx::Jaynes.GenerateContext)(c::typeof(soss_fmi),
                                               fn::typeof(rand),
                                               addr::T,
                                               model::M,
                                               args...) where {T <: Jaynes.Address, M <: Soss.Model}
            kvs = Jaynes.get_sub(ctx.select, addr)
            data = namedtuple(Dict{Symbol, Any}(kvs))
            w, choices = Soss.WeightedSample(call(args...), data)
            score = Soss.logpdf(args, choices)
            Jaynes.add_call!(ctx, addr, Jaynes.SossModelCallSite(choices, score, model, args))
            Jaynes.increment!(ctx, w)
            return choices
        end
    end

    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
