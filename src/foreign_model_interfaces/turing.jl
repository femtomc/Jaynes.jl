macro load_turing_fmi()
    expr = quote
        @info "Loading foreign model interface to \u001b[3m\u001b[34;1mTuring.jl\u001b[0m\n\n          \u001b[34;1mhttps://turing.ml/dev/\n\n "

        import Jaynes: has_top, get_top, has_sub, get_sub, get_score, collect!
        using Turing
    end

    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
