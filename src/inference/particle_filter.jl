mutable struct Particles
    calls::Vector{CallSite}
    lws::Vector{Float64}
    lmle::Float64
end
import Base.length
Base.length(ps::Particles) = length(ps.calls)

function initialize_filter(fn::Function, 
                     args::Tuple;
                     observations = EmptySelection(),
                     num_particles::Int = 5000)
    calls, lnw, lmle = importance_sampling(fn, args; observations = observations, num_samples = num_particles)
    ltw = lmle + log(num_particles)
    lws = lnw .+ ltw
    return Particles(calls, lws, lmle)
end

function filter_step!(ps::Particles,
                      new_args::Tuple;
                      observations = EmptySelection) where T

    num_particles = length(ps)
    update_ctx = Update(Trace(), observations)

    for i in 1:num_particles
        # Run update.
        update_ctx.metadata.tr =  ps.calls[i].trace
        ret = Cassette.overdub(update_ctx, ps.calls[i].fn, new_args...)

        # Store.
        ps.calls[i].args = new_args
        ps.calls[i].ret = ret
        ps.lws[i] = update_ctx.metadata.tr.score
        update_ctx.metadata.constraints = observations
    end
    ltw = lse(ps.lws)
    ps.lmle = ltw - log(num_particles)
end

function filter_step!(ctx::TraceCtx,
                      new_args::Tuple,
                      ps::Particles,
                      proposal::Function,
                      proposal_args::Tuple,
                      observations::ConstrainedSelection) where T

    num_particles = length(ps)
    lws = Vector{Float64}(undef, num_particles)
    rets = Vector{Any}(undef, num_particles)
    prop_ctx = Proposal(Trace())
    update_ctx = Update(Trace(), constraints)

    for i in 1:num_particles
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
        update_ctx.metadata.tr =  ps.calls[i]
        update_ctx.metadata.constraints = constraints
        if isempty(new_args)
            ret = Cassette.overdub(update_ctx, ctx.metadata.fn)
        else
            ret = Cassette.overdub(update_ctx, ctx.metadata.fn, new_args...)
        end

        # Store.
        ps.calls[i] = update_ctx.metadata.tr
        ps.lws[i] = update_ctx.metadata.score - pop_score
        rets[i] = ret
    end

    update_ctx.metadata.ret = rets
    update_ctx.metadata.fn = ctx.metadata.fn
    update_ctx.metadata.args = new_args
    ltw = lse(lws)
    ps.lmle = ltw - log(num_particles)
    return update_ctx, ps
end

# Resample from existing set of particles, mutating the original set.
function resample!(ps::Particles)

    num_particles = length(ps.calls)
    ltw, lnw = nw(ps.lws)
    weights = exp.(lnw)
    selections = rand(Categorical(weights/sum(weights)), num_particles)
    lmle = ltw - log(num_particles)
    calls = map(selections) do ind
        ps.calls[ind]
    end
    ps.calls = calls
    ps.lws = zeros(num_particles)
end

function resample!(ps::Particles,
                   num::Int)

    num_particles = length(ps.calls)
    ltw, lnw = nw(ps.lws)
    weights = exp.(lnw)
    selections = rand(Categorical(weights/sum(weights)), num_particles)
    lmle = ltw - log(num_particles)
    calls = map(selections) do ind
        calls[ind]
    end
    calls = rand(calls, num)
    ps.calls = calls
    ps.lws = zeros(length(ps.calls))
end
