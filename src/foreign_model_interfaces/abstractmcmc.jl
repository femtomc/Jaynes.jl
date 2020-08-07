macro load_abstract_mcmc(expr)
    
    expr = quote
        @info "Constructing kernel interface to \u001b[3m\u001b[34;1mAbstractMCMC.jl\u001b[0m\n\n          \u001b[34;1mhttps://github.com/TuringLang/AbstractMCMC.jl\n\n "

        struct KernelState{K <: Jaynes.UnconstrainedSelection, 
                           P <: Jaynes.Parameters, 
                           F, 
                           A} <: AbstractMCMC.AbstractSamplerState
            sel::K
            params::P
            ker::F
            args::A
        end
        
        struct KernelSampler <: AbstractMCMC.AbstractSampler
            state::KernelState
        end

        function step(rng, call::Function, sampler::KernelSampler; kwargs...)
            _, cl = Jaynes.simulate(call, sampler.state.args...)
            return (cl, sampler.state)
        end
        
        function step(rng, call::Function, sampler::KernelSampler, state; kwargs...)
            prev_cl = state[1]
            new, _ = sampler.ker(sampler.sel, sampler.ps, prev_cl)
            return (new, sampler.state)
        end
    end

    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
