# These functions closely follow the Gen inference library functions. Right now, they are specific to the dynamic DSL here.

# ----------------------------------------------------------------------- #

function importance_sampling(model::Function, 
                             args::Tuple,
                             num_samples::Int)
    trs = Vector{Trace}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    ctx = disablehooks(TraceCtx(metadata = TraceMeta(Trace(), nothing)))
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
    ctx = disablehooks(TraceCtx(metadata = TraceMeta(Trace(), observations)))
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
    ctx = disablehooks(TraceCtx(metadata = TraceMeta(Trace(), observations)))
    for i in 1:num_samples
        # Propose.
        if isempty(proposal_args)
            Cassette.overdub(ctx, proposal)
        else
            Cassette.overdub(ctx, proposal, proposal_args...)
        end

        # Merge proposals and observations.
        prop_score = ctx.metadata.tr.score
        prop_chm = ctx.metadata.tr.chm
        ctx.metadata.constraints = merge(observations, prop_chm)
        reset_keep_constraints!(ctx.metadata)

        # Generate.
        if isempty(args)
            res = Cassette.overdub(ctx, model)
        else
            res = Cassette.overdub(ctx, model, args...)
        end

        # Track score.
        ctx.metadata.tr.func = model
        ctx.metadata.tr.args = args
        ctx.metadata.tr.retval = res
        lws[i] = ctx.metadata.tr.score - prop_score
        trs[i] = ctx.metadata.tr

        # Reset.
        reset_keep_constraints!(ctx.metadata)
        ctx.metadata.constraints = observations
    end
    ltw = lse(lws)
    lmle = ltw - log(num_samples)
    lnw = lws .- ltw
    return trs, lnw, lmle
end
