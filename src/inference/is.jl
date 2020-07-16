function importance_sampling(model::Function, 
                             args::Tuple;
                             observations::ConstrainedSelection = ConstrainedAnywhereSelection(), 
                             num_samples::Int = 5000)
    calls = Vector{HierarchicalCallSite}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    for i in 1:num_samples
        _, calls[i], lws[i] = generate(observations, model, args...)
    end
    ltw = lse(lws)
    lmle = ltw - log(num_samples)
    lnw = lws .- ltw
    return Particles(calls, lws, lmle), lnw
end

function importance_sampling(model::Function, 
                             args::Tuple,
                             proposal::Function,
                             proposal_args::Tuple; 
                             observations::ConstrainedSelection = ConstrainedAnywhereSelection(),
                             num_samples::Int = 5000)
    calls = Vector{HierarchicalCallSite}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    for i in 1:num_samples
        ret, pcall, pw = propose(proposal, proposal_args...)
        select = merge(pcall, observations)
        _, calls[i], lws[i] = generate(select, model, args...)
    end
    ltw = lse(lws)
    lmle = ltw - log(num_samples)
    lnw = lws .- ltw
    return Particles(calls, lws, lmle), lnw
end

# ------------ Documentation ------------ #

@doc(
"""
Samples from the model prior.
```julia
particles, normalized_weights = importance_sampling(model::Function, 
                                                    args::Tuple; 
                                                    observations::ConstrainedSelection = ConstrainedAnywhereSelection(), 
                                                    num_samples::Int = 5000)
```
Samples from a programmer-provided proposal function.
```julia
particles, normalized_weights = importance_sampling(model::Function, 
                                                    args::Tuple, 
                                                    proposal::Function, 
                                                    proposal_args::Tuple; 
                                                    observations::ConstrainedSelection = ConstrainedAnywhereSelection(), 
                                                    num_samples::Int = 5000)
```

Run importance sampling on the posterior over unconstrained addresses and values. Returns an instance of `Particles` and normalized weights.
""", importance_sampling)
