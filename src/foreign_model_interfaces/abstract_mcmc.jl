macro load_abstract_mcmc()

    expr = quote
        @info "Loading kernel interface to \u001b[3m\u001b[34;1mAbstractMCMC.jl\u001b[0m\n\n      \u001b[34;1mhttps://github.com/TuringLang/AbstractMCMC.jl\n\n "
        import AbstractMCMC
        using StatsBase
        import StatsBase: sample
        using MCMCChains
        using StatsPlots

        mutable struct KernelSampler{C <: Jaynes.AddressMap,
                                     T <: Jaynes.AddressMap,
                                     P <: Jaynes.AddressMap,
                                     A,
                                     F,
                                     KA} <: AbstractMCMC.AbstractSampler
            constraints::C
            addresses::Vector
            target::T
            params::P
            args::A
            ker::F
            kwargs::KA
            call::Jaynes.CallSite
            KernelSampler(constraints::C, addrs, tg::T, ps::P, args::A, ker::F, kwargs::KA) where {C, T, P, A, F, KA} = new{C, T, P, A, F, KA}(constraints, addrs, tg, ps, args, ker, kwargs)
        end

        struct BlackBoxWrapper <: AbstractMCMC.AbstractModel
            fn::Function
            BlackBoxWrapper(fn) = new(fn)
        end

        function kernelize(constraints, target::Vector, params, args, primitive; kwargs...)
            KernelSampler(Jaynes.target(constraints), 
                          target, 
                          Jaynes.target(target), 
                          params, 
                          args, 
                          primitive, 
                          kwargs)
        end
        
        function kernelize(constraints, target::Vector, args, primitive; kwargs...)
            KernelSampler(Jaynes.target(constraints), 
                          target, 
                          Jaynes.target(target), 
                          Jaynes.Empty(), 
                          args, 
                          primitive, 
                          kwargs)
        end

        function AbstractMCMC.step(rng, call::BlackBoxWrapper, sampler::KernelSampler; kwargs...)
            _, cl = Jaynes.generate(sampler.constraints, call.fn, sampler.args...)
            (cl, cl)
        end

        function AbstractMCMC.step(rng, call::BlackBoxWrapper, sampler::KernelSampler, state; kwargs...)
            prev_cl = state
            new, acc = sampler.ker(sampler.target, sampler.params, prev_cl; sampler.kwargs...)
            return (new, new)
        end

        function StatsBase.sample(rng, call::Function, sampler::KernelSampler, nsamples::Int; kwargs...)
            sample_arr = StatsBase.sample(rng, BlackBoxWrapper(call), sampler, nsamples; kwargs...)
            chain = Array{Real, 3}(undef, length(sample_arr), length(sampler.addresses), 1)
            for (i, samp) in enumerate(sample_arr)
                for (k, addr) in enumerate(sampler.addresses)
                    chain[i, k, 1] = samp[addr...]
                end
            end
            addrs = map(sampler.addresses) do addr
                String(foldr(=>, addr))
            end
            Chains(chain, addrs)
        end
    end

    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
