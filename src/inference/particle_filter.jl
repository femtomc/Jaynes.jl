mutable struct Particles
    trs::Vector{Trace}
    lws::Vector{Float64}
    lmle::Float64
end
import Base.length
Base.length(ps::Particles) = length(ps.trs)

function initialize_filter(fn::Function, 
                     args::Tuple, 
                     observations::ConstrainedSelection, 
                     num_p::Int)
    ctx, trs, lnw, lmle = importance_sampling(fn, args, observations, num_p)
    ltw = lmle + log(num_p)
    lws = lnw .+ ltw
    return ctx, Particles(trs, lws, lmle)
end

function filter_step!(ctx::TraceCtx,
                      new_args::Tuple,
                      ps::Particles,
                      observations::ConstrainedSelection) where T

    num_p = length(ps)
    rets = Vector{Any}(undef, num_p)
    update_ctx = Update(Trace(), observations)

    for i in 1:num_p
        # Run update.
        update_ctx.metadata.tr =  ps.trs[i]
        if !isempty(new_args)
            ret = Cassette.overdub(update_ctx, ctx.metadata.fn, new_args...)
        else
            ret = Cassette.overdub(update_ctx, ctx.metadata.fn)
        end

        # Store.
        ps.trs[i] = update_ctx.metadata.tr
        ps.lws[i] = update_ctx.metadata.tr.score
        rets[i] = ret
        update_ctx.metadata.constraints = observations
    end

    update_ctx.metadata.ret = rets
    update_ctx.metadata.fn = ctx.metadata.fn
    update_ctx.metadata.args = new_args
    ltw = lse(ps.lws)
    ps.lmle = ltw - log(num_p)
    return update_ctx, ps
end

function filter_step!(ctx::TraceCtx,
                      new_args::Tuple,
                      ps::Particles,
                      proposal::Function,
                      proposal_args::Tuple,
                      observations::ConstrainedSelection) where T

    num_p = length(ps)
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
        update_ctx.metadata.tr =  ps.trs[i]
        update_ctx.metadata.constraints = constraints
        if isempty(new_args)
            ret = Cassette.overdub(update_ctx, ctx.metadata.fn)
        else
            ret = Cassette.overdub(update_ctx, ctx.metadata.fn, new_args...)
        end

        # Store.
        ps.trs[i] = update_ctx.metadata.tr
        ps.lws[i] = update_ctx.metadata.score - pop_score
        rets[i] = ret
    end

    update_ctx.metadata.ret = rets
    update_ctx.metadata.fn = ctx.metadata.fn
    update_ctx.metadata.args = new_args
    ltw = lse(lws)
    ps.lmle = ltw - log(num_p)
    return update_ctx, ps
end

function resample!(ps::Particles)

    num_p = length(ps.trs)
    ltw, lnw = nw(ps.lws)
    weights = exp.(lnw)
    selections = rand(Categorical(weights/sum(weights)), num_p)
    lmle = ltw - log(num_p)
    trs = map(selections) do ind
        ps.trs[ind]
    end
    ps.trs = trs
    ps.lws = zeros(num_p)
end

function resample!(ps::Particles,
                   num::Int)

    num_p = length(ps.trs)
    ltw, lnw = nw(ps.lws)
    weights = exp.(lnw)
    selections = rand(Categorical(weights/sum(weights)), num_p)
    lmle = ltw - log(num_p)
    trs = map(selections) do ind
        trs[ind]
    end
    trs = rand(trs, num)
    ps.trs = trs
    ps.lws = zeros(length(ps.trs))
end
