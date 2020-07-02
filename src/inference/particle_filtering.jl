import Base.length
Base.length(ps::Particles) = length(ps.calls)

function initialize_filter(fn::Function, 
                           args::Tuple,
                           observations::ConstrainedHierarchicalSelection,
                           num_particles::Int)
    return importance_sampling(fn, args; observations = observations, num_samples = num_particles)
end

function filter_step!(ps::Particles,
                      new_args::Tuple,
                      observations::ConstrainedHierarchicalSelection)

    num_particles = length(ps)
    update_ctx = Update(Trace(), observations)

    for i in 1:num_particles
        # Run update.
        update_ctx.tr =  ps.calls[i].trace
        ret = update_ctx(ps.calls[i].fn, new_args...)

        # Store.
        ps.calls[i].args = new_args
        ps.calls[i].ret = ret
        ps.lws[i] = update_ctx.tr.score
        update_ctx.select = observations
    end
    ltw = lse(ps.lws)
    ps.lws = ps.lws .- ltw
    ps.lmle = ltw - log(num_particles)
end

function filter_step!(ps::Particles,
                      new_args::Tuple,
                      proposal::Function,
                      proposal_args::Tuple,
                      observations::ConstrainedHierarchicalSelection)

    num_particles = length(ps)
    lws = Vector{Float64}(undef, num_particles)
    prop_ctx = Proposal(Trace())
    update_ctx = Update(Trace(), constraints)

    for i in 1:num_particles
        # Propose.
        prop_ctx(proposal, ctx.ret[i], proposal_args...)

        # Merge proposals and observations.
        prop_score = prop_ctx.tr.score
        select = merge(prop_ctx.tr, observations)

        # Run update.
        update_ctx.tr =  ps.calls[i]
        update_ctx.select = select
        ret = update_ctx(ps.calls[i].fn, new_args...)

        # Store.
        ps.calls[i].args = new_args
        ps.calls[i].ret = ret
        ps.lws[i] = update_ctx.score - pop_score
    end

    ltw = lse(ps.lws)
    ps.lws = ps.lws .- ltw
    ps.lmle = ltw - log(num_particles)
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
        ps.calls[ind]
    end
    calls = rand(calls, num)
    ps.calls = calls
    ps.lws = zeros(length(ps.calls))
end
