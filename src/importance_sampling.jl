# These functions closely follow the Gen inference library functions. Right now, they are specific to the dynamic DSL here.

# ----------------------------------------------------------------------- #

function importance_sampling(model::Function, 
                             args::Tuple,
                             num_samples::Int)
    trs = Vector{Trace}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    ctx = disablehooks(TraceCtx(metadata = Trace()))
    for i in 1:num_samples
        if isempty(args)
            res = Cassette.overdub(ctx, model)
        else
            res = Cassette.overdub(ctx, model, args...)
        end
        ctx.metadata.func = model
        ctx.metadata.args = args
        ctx.metadata.retval = res
        lws[i] = ctx.metadata.score
        trs[i] = ctx.metadata
        ctx = similarcontext(ctx, metadata = Trace())
    end
    ltw = lse(lws)
    lmle = ltw - log(num_samples)
    lnw = lws .- ltw
    return trs, lnw, lmle
end

function importance_sampling(model::Function, 
                             args::Tuple,
                             observations::Dict{Address, Union{Int64, Float64}},
                             num_samples::Int)
    trs = Vector{Trace}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    ctx = disablehooks(TraceCtx(metadata = Trace(observations)))
    for i in 1:num_samples
        if isempty(args)
            res = Cassette.overdub(ctx, model)
        else
            res = Cassette.overdub(ctx, model, args...)
        end
        ctx.metadata.func = model
        ctx.metadata.args = args
        ctx.metadata.retval = res
        lws[i] = ctx.metadata.score
        trs[i] = ctx.metadata
        ctx = similarcontext(ctx, metadata = Trace(observations))
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
                             observations::Dict{Address, Union{Int64, Float64}},
                             num_samples::Int)
    trs = Vector{Trace}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    prop_ctx = disablehooks(TraceCtx(metadata = Trace(observations)))
    model_ctx = disablehooks(TraceCtx(metadata = Trace(observations)))
    for i in 1:num_samples
        # Propose.
        if isempty(proposal_args)
            Cassette.overdub(prop_ctx, proposal)
        else
            Cassette.overdub(prop_ctx, proposal, proposal_args...)
        end

        # Merge proposals and observations.
        prop_score = prop_ctx.metadata.score
        prop_chm = prop_ctx.metadata.chm
        constraints = merge(observations, prop_chm)
        prop_ctx = similarcontext(prop_ctx, metadata = Trace(observations))

        # Generate.
        if isempty(args)
            res = Cassette.overdub(model_ctx, model)
        else
            res = Cassette.overdub(model_ctx, model, args...)
        end

        # Track score.
        model_ctx.metadata.func = model
        model_ctx.metadata.args = args
        model_ctx.metadata.retval = res
        lws[i] = model_ctx.metadata.score - prop_score
        trs[i] = model_ctx.metadata
        model_ctx = similarcontext(model_ctx, metadata = Trace(observations))
    end
    ltw = lse(lws)
    lmle = ltw - log(num_samples)
    lnw = lws .- ltw
    return trs, lnw, lmle
end
