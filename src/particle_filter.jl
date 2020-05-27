# Currently assumes that the args to the next proposal are stored in the ret field of the context.
function filter_step!(ctx::TraceCtx{M},
                      trs::Vector{Trace}, 
                      observations::Dict{Address, T}) where {T, M <: UnconstrainedGenerateMeta}
    num_p = length(trs)
    lws = Vector{Float64}(undef, num_p)
    rets = Vector{Any}(undef, num_p)
    update_ctx = Update(Trace(), observations)

    for i in 1:num_p
        # Run update.
        update_ctx.metadata.tr =  trs[i]
        if !isempty(args)
            ret = Cassette.overdub(update_ctx, ctx.func, ctx.metadata.ret[i])
        else

            ret = Cassette.overdub(update_ctx, ctx.func)
        end

        # Store.
        trs[i] = new_tr
        lws[i] = new_score
        rets[i] = update_ctx.metadata.ret
    end

    update_ctx.metadata.ret = rets
    update.ctx.metadata.func = ctx.func
    update_ctx.metadata.args = args
    return update_ctx, trs, lws, rets
end

# Currently assumes that the args to the next proposal are stored in the ret field of the context.
function filter_step!(ctx::TraceCtx{M},
                      trs::Vector{Trace}, 
                      proposal::Function,
                      proposal_args::Tuple,
                      observations::Dict{Address, T}) where {T, M <: UnconstrainedGenerateMeta}
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
        if isempty(args)
            ret = Cassette.overdub(update_ctx, ctx.func)
        else
            ret = Cassette.overdub(update_ctx, ctx.func args...)
        end

        # Store.
        trs[i] = update_ctx.metadata.tr
        lws[i] = update_ctx.metadata.score - pop_score
        rets[i] = ret
    end

    update_ctx.metadata.ret = rets
    update_ctx.metadata.func = ctx.func
    update_ctx.metadata.args = args
    return ctx, trs, lws, rets
end

function resample!(trs::Vector{Trace}, 
                   lws::Vector{Float64})
    num_p = length(trs)
    ltw, lnw = nw(lws)
    weights = exp.(lnw)
    selections = rand(Categorical(weights/sum(weights)), 1:num_p)
    lmle += ltw - log(num_p)
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
    selections = rand(Categorical(weights/sum(weights)), 1:num_p)
    lmle += ltw - log(num_p)
    trs = map(selections) do ind
        trs[ind]
    end
    lws = map(lws) do k
        0.0
    end
    trs = rand(trs, num)
    return trs, lws, lmle
end
