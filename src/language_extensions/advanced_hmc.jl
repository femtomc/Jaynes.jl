macro load_advanced_hmc()

    expr = quote
        @info "Loading interface to \u001b[3m\u001b[34;1mAdvancedHMC.jl\u001b[0m\n\n      \u001b[34;1mhttps://github.com/TuringLang/AdvancedHMC.jl\n\n "
        try
            using AdvancedHMC
            import AdvancedHMC: sample
            using MCMCChains
        catch
            error("This interface requires that your environment has the following dependencies:\n\n AdvancedHMC\n MCMCChains\n")
        end

        function π(vals, target, cl)
            target_values = Jaynes.target(target, vals)
            ret, u_cl, w, _ = Jaynes.update(target_values, cl)
            Jaynes.get_score(u_cl)
        end
        
        function ∇π(vals, target::K, cl) where K <: Jaynes.AddressMap{Jaynes.Value}
            target_values = Jaynes.target(target, vals)
            ret, u_cl, w, _ = Jaynes.update(target_values, cl)
            sel = select(target)
            _, _, choice_grads = Jaynes.get_choice_gradients(sel, u_cl, 1.0)
            grads = Jaynes.array(choice_grads, Float64)
            Jaynes.get_score(u_cl), grads
        end
        
        function π(vals, target, cl)
            target_values = Jaynes.target(target, vals)
            ret, u_cl, w, _ = Jaynes.update(target_values, cl)
            Jaynes.get_score(u_cl)
        end
        
        function ∇π(vals, target::K, cl) where K <: Jaynes.Target
            target_values = Jaynes.target(target, vals)
            ret, u_cl, w, _ = Jaynes.update(target_values, cl)
            _, _, choice_grads = Jaynes.get_choice_gradients(target, u_cl, 1.0)
            grads = Jaynes.array(choice_grads, Float64)
            Jaynes.get_score(u_cl), grads
        end

        function advanced_hmc(initial_θ::P,
                              obs::C, 
                              model::Function,
                              args...; 
                              n_samples = 2000,
                              n_adapts = 1000,
                              metric = DiagEuclideanMetric(Jaynes.value_length(initial_θ))) where {T <: Jaynes.AddressMap, P <: Jaynes.AddressMap, C <: Jaynes.AddressMap}

            ret, cl, w = Jaynes.generate(obs, model, args...)
            hamiltonian = Hamiltonian(metric, 
                                      vals -> π(vals, initial_θ, cl), 
                                      vals -> ∇π(vals, initial_θ, cl))
            initial_ϵ = find_good_stepsize(hamiltonian, 
                                           array(initial_θ, Float64))
            integrator = Leapfrog(initial_ϵ)
            proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
            adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), 
                                     StepSizeAdaptor(0.8, integrator))
            samples, stats = sample(hamiltonian, 
                                    proposal, 
                                    array(initial_θ, Float64),
                                    n_samples, 
                                    adaptor, 
                                    n_adapts; 
                                    progress=true)
            Chains(samples), stats
        end

        const ahmc = advanced_hmc
    end

    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
