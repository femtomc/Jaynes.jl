macro load_flux_fmi()
    @info "Loading differentiable compatibility to \u001b[3m\u001b[34;1mFlux.jl\u001b[0m\n\n          \u001b[34;1mhttps://github.com/FluxML/Flux.jl\n "
    expr = quote
        using Zygote
        using Flux
        using Flux: Chain, Dense
    end

    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
