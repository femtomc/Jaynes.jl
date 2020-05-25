# These functions closely follow the Gen inference library functions. Right now, they are specific to the dynamic DSL here.

# TODO: extract core of routines into inference_interfaces.

# ----------------------------------------------------------------------- #

function importance_sampling(model::Function, 
                             args::Tuple,
                             num_samples::Int)
    trs = Vector{Trace}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    ctx = disablehooks(TraceCtx(metadata = UnconstrainedGenerateMeta(Trace())))
    for i in 1:num_samples
        if isempty(args)
            res = Cassette.overdub(ctx, model)
        else
            res = Cassette.overdub(ctx, model, args...)
        end
        ctx.metadata.tr.func = model
        ctx.metadata.tr.args = args
        ctx.metadata.tr.retval = res
        lws[i] = ctx.metadata.tr.score
        trs[i] = ctx.metadata.tr
        reset_keep_constraints!(ctx.metadata)
    end
    ltw = lse(lws)
    lmle = ltw - log(num_samples)
    lnw = lws .- ltw
    return trs, lnw, lmle
end

function importance_sampling(model::Function, 
                             args::Tuple,
                             observations::Dict{Address, T},
                             num_samples::Int) where T
    trs = Vector{Trace}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    ctx = disablehooks(TraceCtx(metadata = GenerateMeta(Trace(), observations)))
    for i in 1:num_samples
        if isempty(args)
            res = Cassette.overdub(ctx, model)
        else
            res = Cassette.overdub(ctx, model, args...)
        end
        ctx.metadata.tr.func = model
        ctx.metadata.tr.args = args
        ctx.metadata.tr.retval = res
        lws[i] = ctx.metadata.tr.score
        trs[i] = ctx.metadata.tr
        reset_keep_constraints!(ctx.metadata)
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
                             observations::Dict{Address, T},
                             num_samples::Int) where T
    trs = Vector{Trace}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
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
            res = Cassette.overdub(model_ctx, model)
        else
            res = Cassette.overdub(model_ctx, model, args...)
        end

        # Track score.
        model_ctx.metadata.tr.func = model
        model_ctx.metadata.tr.args = args
        model_ctx.metadata.tr.retval = res
        lws[i] = model_ctx.metadata.tr.score - prop_score
        trs[i] = model_ctx.metadata.tr

        # Reset.
        reset_keep_constraints!(model_ctx.metadata)
        reset_keep_constraints!(prop_ctx.metadata)
    end
    ltw = lse(lws)
    lmle = ltw - log(num_samples)
    lnw = lws .- ltw
    return trs, lnw, lmle
end
