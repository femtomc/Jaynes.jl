function importance_sampling(model::Function, 
                             args::Tuple;
                             observations::ConstrainedSelection = ConstrainedAnywhereSelection(), 
                             num_samples::Int = 5000)
    calls = Vector{BlackBoxCallSite}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    ctx = Generate(Trace(), observations)
    for i in 1:num_samples
        ret = ctx(model, args...)
        lws[i] = ctx.tr.score
        calls[i] = BlackBoxCallSite(ctx.tr, 
                            model, 
                            args,
                            ret)
        ctx.tr = Trace()
    end
    ltw = lse(lws)
    lmle = ltw - log(num_samples)
    lnw = lws .- ltw
    return Particles(calls, lnw, lmle)
end

function importance_sampling(model::Function, 
                             args::Tuple,
                             proposal::Function,
                             proposal_args::Tuple; 
                             observations::ConstrainedSelection = ConstrainedAnywhereSelection(),
                             num_samples::Int = 5000) where T
    calls = Vector{BlackBoxCallSite}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    prop_ctx = Propose(Trace())
    model_ctx = Generate(Trace(), observations)
    for i in 1:num_samples
        # Propose.
        prop_ctx(proposal, proposal_args...)

        # Merge proposals and observations.
        prop_score = prop_ctx.tr.score
        select = merge(prop_ctx.tr, observations)
        model_ctx.select = select

        # Generate.
        ret = model_ctx(model, args...)
        !compare(select, model_ctx.visited) && error("ProposalError: support error - not all constraints provided by merge of proposal and observations were visited.")

        # Track.
        calls[i] = BlackBoxCallSite(model_ctx.tr, 
                            model, 
                            args,
                            ret)
        lws[i] = model_ctx.tr.score - prop_score

        # Reset.
        model_ctx.tr = Trace()
        prop_ctx.tr = Trace()
    end
    ltw = lse(lws)
    lmle = ltw - log(num_samples)
    lnw = lws .- ltw
    return Particles(calls, lnw, lmle)
end
