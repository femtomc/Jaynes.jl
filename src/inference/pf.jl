import Base.length
Base.length(ps::Particles) = length(ps.calls)

function initialize_filter(observations::K,
                           num_particles::Int,
                           fn::Function, 
                           args::Tuple) where K <: ConstrainedSelection
    ps, _ = importance_sampling(observations, num_particles, fn, args)
    return ps
end

function filter_step!(observations::K,
                      ps::Particles,
                      argdiffs::D,
                      new_args::Tuple) where {K <: ConstrainedSelection, D <: Diff}
    num_particles = length(ps)
    Threads.@threads for i in 1:num_particles
        _, ps.calls[i], uw, _, _ = update(observations, ps.calls[i], argdiffs, new_args...)
        ps.lws[i] += uw
    end
    ltw = lse(ps.lws)
    ps.lmle = ltw - log(num_particles)
end

function filter_step!(observations::K,
                      ps::Particles,
                      argdiffs::D,
                      new_args::Tuple,
                      proposal::Function,
                      proposal_args::Tuple) where {K <: ConstrainedSelection, D <: Diff}
    num_particles = length(ps)
    Threads.@threads for i in 1:num_particles
        _, p_cl, p_w = propose(proposal, ps.calls[i], proposal_args...)
        sel = selection(p_cl)
        merge!(sel, observations)
        _, ps.calls[i], u_w, _, _ = update(sel, ps.calls[i], argdiffs, new_args...)
        ps.lws[i] += u_w - p_w
    end
    ltw = lse(ps.lws)
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
