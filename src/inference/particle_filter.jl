function filter_step!(ctx::TraceCtx,
                      new_args::Tuple,
                      trs::Vector{Trace}, 
                      observations::ConstrainedSelection) where T

    num_p = length(trs)
    lws = Vector{Float64}(undef, num_p)
    rets = Vector{Any}(undef, num_p)
    update_ctx = Update(Trace(), observations)

    for i in 1:num_p
        # Run update.
        update_ctx.metadata.tr =  trs[i]
        if !isempty(new_args)
            ret = Cassette.overdub(update_ctx, ctx.metadata.fn, new_args...)
        else
            ret = Cassette.overdub(update_ctx, ctx.metadata.fn)
        end

        # Store.
        trs[i] = update_ctx.metadata.tr
        lws[i] = update_ctx.metadata.tr.score
        rets[i] = ret
        update_ctx.metadata.constraints = observations
    end

    update_ctx.metadata.ret = rets
    update_ctx.metadata.fn = ctx.metadata.fn
    update_ctx.metadata.args = new_args
    ltw = lse(lws)
    lmle = ltw - log(num_p)
    lnw = lws .- ltw
    return update_ctx, trs, lnw, lmle
end

function filter_step!(ctx::TraceCtx,
                      new_args::Tuple,
                      trs::Vector{Trace}, 
                      proposal::Function,
                      proposal_args::Tuple,
                      observations::ConstrainedSelection) where T

    num_p = length(trs)
    lws = Vector{Float64}(undef, num_p)
    rets = Vector{Any}(undef, num_p)
    prop_ctx = Proposal(Trace())
    update_ctx = Update(Trace(), constraints)

    for i in 1:num_p
        # Propose.
        if isempty(proposal_args)
            Cassette.overdub(prop_ctx, proposal)
        else
            Cassette.overdub(prop_ctx, proposal, (ctx.metadata.ret[i], proposal_args...))
        end

        # Merge proposals and observations.
        prop_score = prop_ctx.metadata.tr.score
        prop_chm = prop_ctx.metadata.tr.chm
        constraints = merge(observations, prop_chm)

        # Run update.
        update_ctx.metadata.tr =  trs[i]
        update_ctx.metadata.constraints = constraints
        if isempty(new_args)
            ret = Cassette.overdub(update_ctx, ctx.metadata.fn)
        else
            ret = Cassette.overdub(update_ctx, ctx.metadata.fn, new_args...)
        end

        # Store.
        trs[i] = update_ctx.metadata.tr
        lws[i] = update_ctx.metadata.score - pop_score
        rets[i] = ret
    end

    update_ctx.metadata.ret = rets
    update_ctx.metadata.fn = ctx.metadata.fn
    update_ctx.metadata.args = new_args
    ltw = lse(lws)
    lmle = ltw - log(num_p)
    lnw = lws .- ltw
    return update_ctx, trs, lnw, lmle
end

function resample!(trs::Vector{Trace}, 
                   lws::Vector{Float64})

    num_p = length(trs)
    ltw, lnw = nw(lws)
    weights = exp.(lnw)
    selections = rand(Categorical(weights/sum(weights)), num_p)
    lmle = ltw - log(num_p)
    trs = map(selections) do ind
        trs[ind]
    end
    lws = map(lws) do k
        0.0
    end
    return trs, lws, lmle
end

function resample!(trs::Vector{Trace}, 
                  lws::Vector{Float64},
                  num::Int)

    num_p = length(trs)
    ltw, lnw = nw(lws)
    weights = exp.(lnw)
    selections = rand(Categorical(weights/sum(weights)), num_p)
    lmle = ltw - log(num_p)
    trs = map(selections) do ind
        trs[ind]
    end
    lws = map(lws) do k
        0.0
    end
    trs = rand(trs, num)
    return trs, lws, lmle
end
