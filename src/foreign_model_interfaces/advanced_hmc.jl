macro load_advanced_hmc()

    expr = quote
        @info "Loading interface to \u001b[3m\u001b[34;1mAdvancedMCMC.jl\u001b[0m\n\n      \u001b[34;1mhttps://github.com/TuringLang/AdvancedHMC.jl\n\n "
        try
            using AdvancedHMC
            import AdvancedHMC: sample
            using MCMCChains
        catch
            error("This interface requires that your environment has the following dependencies:\n\n StatsBase\n MCMCChains\n StatsPlots\n AbstractMCMC\n")
        end

        struct AdvancedHMCRecipe{K <: AddressMap}
            target::K
            n_samples::Int
            n_adapts::Int
            initial_θ::Vector{Float64}
            metric
            AdvancedHMCRecipe(target::K, initial_θ::Vector{Float64}) = new(target, 2000, 1000, initial_θ, DiagEuclideanMetric(length(initial_θ)))
        end

        function ∇π(vals, target, cl)
            target_values = target(target, vals)
            ret, u_cl, w, _ = update(target_values, cl)
            _, _, choice_grads = get_choice_gradients(target, u_cl, 1.0)
            grads = array(choice_grads, Float64)
            get_score(u_cl), grads
        end

        function π(vals, target, cl)
            target_values = target(target, vals)
            ret, u_cl, w, _ = update(target_values, cl)
            get_score(u_cl)
        end

        function sample(recipe::HMCRecipe, constraints, model, args...)
            ret, cl, w = generate(constraints, model, args...)
            hamiltonian = Hamiltonian(recipe.metric, 
                                      vals -> π(vals, recipe.target, cl), 
                                      vals -> ∇π(vals, recipe.target, cl))
            initial_ϵ = find_good_stepsize(hamiltonian, 
                                           recipe.initial_θ)
            integrator = Leapfrog(initial_ϵ)
            proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(recipe.integrator)
            adaptor = StanHMCAdaptor(MassMatrixAdaptor(recipe.metric), 
                                     StepSizeAdaptor(0.8, recipe.integrator))
            samples, stats = sample(hamiltonian, 
                                    proposal, 
                                    recipe.initial_θ, 
                                    recipe.n_samples, 
                                    recipe.adaptor, 
                                    recipe.n_adapts; 
                                    progress=true)
            Chains(samples), stats
        end
    end

    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
