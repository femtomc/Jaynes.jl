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
end

function check_ess_resample!(ps::Particles)
    num_particles = length(ps.calls)
    ltw, lnw = nw(ps.lws)
    if ess(lnw) < length(ps) / 2
        weights = exp.(lnw)
        ps.lmle += ltw - log(num_particles)
        selections = rand(Categorical(weights/sum(weights)), num_particles)
        calls = map(selections) do ind
            ps.calls[ind]
        end
        ps.calls = calls
        ps.lws = zeros(num_particles)
        return true
    end
    return false
end

function get_lmle(ps::Particles)
    return ps.lmle + lse(ps.lws) - log(length(ps))
end

# Particle filter
function particle_filter(observations::Dict{Int, K},
                         steps::Int,
                         num_particles::Int,
                         model::Function,
                         argdiffs::Dict{Int, D},
                         args::Dict{Int, T}) where {K <: ConstrainedSelection, D <: Diff, T <: Tuple}
    local ps
    for i in 0 : steps
        !haskey(args, i) && error("ParticleFilter: user must specify input args for each time step, including the initialization step (at time 0). Missing $(i).")
        i > 0 && !haskey(argdiffs, i) && error("ParticleFilter: user must specify argdiffs for each time step after initialization. Missing $(i).")
    end
    if haskey(observations, 0)
        ps = initialize_filter(observations[0], num_particles, model, args[0])
    else
        ps = initialize_filter(selection(), num_particles, model, args[0])
    end
    for i in 1 : steps
        if haskey(observations, i)
            filter_step!(observations[i], ps, argdiffs[i], args[i])
        else
            filter_step!(selection(), ps, argdiffs[i], args[i])
        end
        check_ess_resample!(ps)
    end
    return ps
end

function particle_filter(observations::Dict{Int, K},
                         steps::Int,
                         num_particles::Int,
                         model::Function,
                         argdiffs::Dict{Int, D},
                         args::Dict{Int, T},
                         proposals::Dict{Int, Function},
                         proposal_args::Dict{Int, Tuple}) where {K <: ConstrainedSelection, D <: Diff, T <: Tuple}
    local ps
    for i in 0 : steps
        !haskey(args, i) && error("ParticleFilter: user must specify input args for each time step, including the initialization step (at time 0).")
        !haskey(argdiffs, i) && error("ParticleFilter: user must specify argdiffs for each time step, including the initialization step (at time 0).")
        !haskey(proposals, i) && error("ParticleFilter: user must specify proposal for each time step, including the initialization step (at time 0).")
        !haskey(proposal_args, i) && error("ParticleFilter: user must specify input proposal args for each time step, including the initialization step (at time 0).")
    end
    if haskey(observations, 0)
        ps = initialize_filter(observations[0], num_particles, model, args[0])
    else
        ps = initialize_filter(selection(), num_particles, model, args[0])
    end
    for i in 1 : steps
        if haskey(observations, i)
            filter_step!(observations[i], ps, argdiffs[i], args[i], proposals[i], proposal_args[i])
        else
            filter_step!(selection(), ps, argdiffs[i], args[i], proposals[i], proposal_args[i])
        end
        check_ess_resample!(ps)
    end
    return ps
end
