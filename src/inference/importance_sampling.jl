# These functions closely follow the implementation of the Gen inference library functions. Right now, they are specific to the dynamic DSL here.

# ----------------------------------------------------------------------- #

function importance_sampling(model::Function, 
                             args::Tuple;
                             observations::ConstrainedSelection = ConstrainedAnywhereSelection(), 
                             num_samples::Int = 5000)
    calls = Vector{CallSite}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    ctx = Generate(Trace(), observations)
    for i in 1:num_samples
        ret = Cassette.overdub(ctx, model, args...)
        lws[i] = ctx.metadata.tr.score
        calls[i] = CallSite(ctx.metadata.tr, 
                            model, 
                            args,
                            ret)
        reset!(ctx)
    end
    ltw = lse(lws)
    lmle = ltw - log(num_samples)
    lnw = lws .- ltw
    return calls, lnw, lmle
end

function importance_sampling(model::Function, 
                             args::Tuple,
                             proposal::Function,
                             proposal_args::Tuple; 
                             observations::ConstrainedSelection = ConstrainedAnywhereSelection(),
                             num_samples::Int = 5000) where T
    calls = Vector{CallSite}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    prop_ctx = Propose(Trace())
    model_ctx = Generate(Trace(), observations)
    for i in 1:num_samples
        # Propose.
        Cassette.overdub(prop_ctx, proposal, proposal_args...)

        # Merge proposals and observations.
        prop_score = prop_ctx.metadata.tr.score
        select = merge(prop_ctx.metadata.tr, observations)
        model_ctx.metadata.select = select

        # Generate.
        ret = Cassette.overdub(model_ctx, model, args...)
        !compare(select, model_ctx.metadata.visited) && error("ProposalError: support error - not all constraints provided by merge of proposal and observations were visited.")

        # Track.
        calls[i] = CallSite(model_ctx.metadata.tr, 
                            model, 
                            args,
                            ret)
        lws[i] = model_ctx.metadata.tr.score - prop_score

        # Reset.
        reset!(model_ctx)
        reset!(prop_ctx)
    end
    ltw = lse(lws)
    lmle = ltw - log(num_samples)
    lnw = lws .- ltw
    return calls, lnw, lmle
end
