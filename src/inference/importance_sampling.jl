# These functions closely follow the implementation of the Gen inference library functions. Right now, they are specific to the dynamic DSL here.

# ----------------------------------------------------------------------- #

function importance_sampling(model::Function, 
                             args::Tuple;
                             observations::ConstrainedSelection = ConstrainedAnywhereSelection(), 
                             num_samples::Int = 5000)
    calls = Vector{CallSite}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    ctx = Generate(ignore_pass, Trace(), observations)
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
    prop_ctx = Propose(ignore_pass, Trace())
    model_ctx = Generate(ignore_pass, Trace(), observations)
    for i in 1:num_samples
        # Propose.
        if isempty(proposal_args)
            Cassette.overdub(prop_ctx, proposal)
        else
            Cassette.overdub(prop_ctx, proposal, proposal_args...)
        end

        # Merge proposals and observations.
        prop_score = prop_ctx.metadata.tr.score
        select = merge(prop_ctx.metadata.tr, observations)
        model_ctx.metadata.select = select

        # Generate.
        if isempty(args)
            ret = Cassette.overdub(model_ctx, model)
        else
            ret = Cassette.overdub(model_ctx, model, args...)
        end

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
