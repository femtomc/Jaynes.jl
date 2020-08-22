macro load_pyro_fmi()
    @info "Loading foreign model interface to \u001b[3m\u001b[34;1mPyro\u001b[0m\n\n          \u001b[34;1mhttps://pyro.ai/\n\nThis interface is provided by Pyrox.jl."
    expr = quote
        using Pyrox
    end

    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
