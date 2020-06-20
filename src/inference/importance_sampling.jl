# These functions closely follow the implementation of the Gen inference library functions. Right now, they are specific to the dynamic DSL here.

# ----------------------------------------------------------------------- #

function importance_sampling(model::Function, 
                             args::Tuple;
                             observations = EmptySelection(),
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
        reset_keep_constraints!(ctx)
    end
    ltw = lse(lws)
    lmle = ltw - log(num_samples)
    lnw = lws .- ltw
    return calls, lnw, lmle
end

function importance_sampling(model::Function, 
                             args::Tuple,
                             proposal::Function,
                             proposal_args::Tuple,
                             observations::ConstrainedHierarchicalSelection,
                             num_samples::Int) where T
    trs = Vector{Trace}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    rets = Vector{Any}(undef, num_samples)
    prop_ctx = disablehooks(TraceCtx(metadata = ProposalMeta(Trace())))
    model_ctx = disablehooks(TraceCtx(metadata = GenerateMeta(Trace(), observations)))
    for i in 1:num_samples
        # Propose.
        if isempty(proposal_args)
            Cassette.overdub(prop_ctx, proposal)
        else
            Cassette.overdub(prop_ctx, proposal, proposal_args...)
        end

        # Merge proposals and observations.
        prop_score = prop_ctx.metadata.tr.score
        prop_chm = prop_ctx.metadata.tr.chm
        constraints = merge(observations, prop_chm)
        model_ctx.metadata.constraints = constraints

        # Generate.
        if isempty(args)
            ret = Cassette.overdub(model_ctx, model)
        else
            ret = Cassette.overdub(model_ctx, model, args...)
        end

        # Track.
        rets[i] = ret
        trs[i] = model_ctx.metadata.tr
        lws[i] = model_ctx.metadata.tr.score - prop_score

        # Reset.
        reset_keep_constraints!(model_ctx)
        reset_keep_constraints!(prop_ctx)
    end
    model_ctx.metadata.fn = model
    model_ctx.metadata.args = args
    model_ctx.metadata.ret = rets
    ltw = lse(lws)
    lmle = ltw - log(num_samples)
    lnw = lws .- ltw
    return model_ctx, trs, lnw, lmle
end
