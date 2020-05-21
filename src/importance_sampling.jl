function importance_sampling(model::Function, 
                             args::Tuple,
                             num_samples::Int)
    trs = Vector{Trace}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    for i in 1:num_samples
        ctx = TraceCtx(metadata = Trace())
        if isempty(args)
            Cassette.overdub(ctx, model)
        else
            Cassette.overdub(ctx, model, args...)
        end

        lws[i] = ctx.metadata.score
        trs[i] = ctx.metadata
    end
    ltw = lse(lws)
    lmle = ltw - log(num_samples)
    lnw = lws .- ltw
    return trs, lnw, lmle
end

function importance_sampling(model::Function, 
                             args::Tuple,
                             observations::Dict{Address, Real},
                             num_samples::Int)
    trs = Vector{Trace}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    for i in 1:num_samples
        ctx = TraceCtx(metadata = Trace(observations))
        if isempty(args)
            Cassette.overdub(ctx, model)
        else
            Cassette.overdub(ctx, model, args...)
        end
        lws[i] = ctx.metadata.score
        trs[i] = ctx.metadata
    end
    ltw = lse(lws)
    lmle = ltw - log(num_samples)
    lnw = lws .- ltw
    return trs, lnw, lmle
end

function importance_sampling(model::Function, 
                             args::Tuple,
                             proposal::Function,
                             proposal_args::Tuple,
                             observations::Dict{Address, Real},
                             num_samples::Int)
    trs = Vector{Trace}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    for i in 1:num_samples
        # Propose.
        prop_ctx = TraceCtx(metadata = Trace(observations))
        if isempty(proposal_args)
            Cassette.overdub(prop_ctx, proposal)
        else
            Cassette.overdub(prop_ctx, proposal, proposal_args...)
        end

        # Merge proposals and observations.
        prop_score = prop_ctx.metadata.score
        prop_chm = prop_ctx.metadata.chm
        constraints = merge(observations, prop_chm)

        # New context.
        model_ctx = TraceCtx(metadata = Trace(constraints))

        # Generate.
        if isempty(args)
            Cassette.overdub(model_ctx, model)
        else
            Cassette.overdub(model_ctx, model, args...)
        end

        # Track score.
        lws[i] = model_ctx.metadata.score - prop_score
        trs[i] = model_ctx.metadata
    end
    ltw = lse(lws)
    lmle = ltw - log(num_samples)
    lnw = lws .- ltw
    return trs, lnw, lmle
end
