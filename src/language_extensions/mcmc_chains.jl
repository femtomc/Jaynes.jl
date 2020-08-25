macro load_chains()

    expr = quote
        @info "Loading diagnostic interface to \u001b[3m\u001b[34;1mMCMCChains.jl\u001b[0m\n\n      \u001b[34;1mhttps://github.com/TuringLang/MCMCChains.jl\n\n "
        try
            using MCMCChains
            using StatsPlots
            using Random
        catch
            error("This interface requires that your environment has the following dependencies:\n\n MCMCChains\n GR\n StatsPlots\n")
        end
        
        function chain(target::T, calls::Vector{C}) where {T <: Jaynes.Target, C, N}
            addrs, _, _ = flatten(target)
            chains = Array{Float64, 3}(undef, (length(calls), length(addrs), 1))
            for (k, addr) in enumerate(addrs)
                for (j, cl) in enumerate(calls)
                    chains[j, k, 1] = Jaynes.has_value(cl, addr) ? Jaynes.get_value(cl, addr) : missing
                end
            end
            Chains(chains, map(addrs) do addr
                       foldr(=>, addr)
                   end)
        end

        function chain(target::T, calls::Array{C, N}) where {T <: Jaynes.Target, C, N}
            addrs, _, _ = flatten(target)
            chains = Array{Float64, 3}(undef, (length(calls), length(addrs), N))
            for i in 1 : N
                for (k, addr) in enumerate(addrs)
                    for (j, cl) in enumerate(calls)
                        chains[j, k, i] = Jaynes.has_value(cl, addr) ? Jaynes.get_value(cl, addr) : missing
                    end
                end
            end
            Chains(chains, map(addrs) do addr
                       foldr(=>, addr)
                   end)
        end
    end

    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
