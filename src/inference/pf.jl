import Base.length
Base.length(ps::Particles) = length(ps.calls)

function initialize_filter(observations::K,
                           num_particles::Int,
                           fn::Function, 
                           args::Tuple) where K <: AddressMap
    ps, _ = importance_sampling(observations, num_particles, fn, args)
    return ps
end

function initialize_filter(observations::K,
                           ps::P,
                           num_particles::Int,
                           fn::Function, 
                           args::Tuple) where {K <: AddressMap, P <: AddressMap}
    ps, _ = importance_sampling(observations, ps, num_particles, fn, args)
    return ps
end

function filter_step!(observations::K,
                      ps::Particles) where {K <: AddressMap, D <: Diff}
    num_particles = length(ps)
    Threads.@threads for i in 1:num_particles
        _, ps.calls[i], uw, _, _ = update(observations, ps.calls[i])
        ps.lws[i] += uw
    end
end

function filter_step!(observations::K,
                      ps::Particles,
                      argdiffs::D,
                      new_args::Tuple) where {K <: AddressMap, D <: Diff}
    num_particles = length(ps)
    Threads.@threads for i in 1:num_particles
        _, ps.calls[i], uw, _, _ = update(observations, ps.calls[i], argdiffs, new_args...)
        ps.lws[i] += uw
    end
end

function filter_step!(observations::K,
                      params::P,
                      ps::Particles,
                      argdiffs::D,
                      new_args::Tuple) where {K <: AddressMap, P <: AddressMap, D <: Diff}
    num_particles = length(ps)
    Threads.@threads for i in 1:num_particles
        _, ps.calls[i], uw, _, _ = update(observations, params, ps.calls[i], argdiffs, new_args...)
        ps.lws[i] += uw
    end
end

function filter_step!(observations::K,
                      ps::Particles,
                      proposal::Function,
                      proposal_args::Tuple) where {K <: AddressMap, D <: Diff}
    num_particles = length(ps)
    Threads.@threads for i in 1:num_particles
        _, p_cl, p_w = propose(proposal, ps.calls[i], proposal_args...)
        sel = selection(p_cl)
        merge!(sel, observations)
        _, ps.calls[i], u_w, _, _ = update(sel, ps.calls[i])
        ps.lws[i] += u_w - p_w
    end
end

function filter_step!(observations::K,
                      params::P,
                      ps::Particles,
                      proposal::Function,
                      proposal_args::Tuple) where {K <: AddressMap, P <: AddressMap, D <: Diff}
    num_particles = length(ps)
    Threads.@threads for i in 1:num_particles
        _, p_cl, p_w = propose(proposal, ps.calls[i], proposal_args...)
        sel = selection(p_cl)
        merge!(sel, observations)
        _, ps.calls[i], u_w, _, _ = update(sel, params, ps.calls[i])
        ps.lws[i] += u_w - p_w
    end
end

function filter_step!(observations::K,
                      ps::Particles,
                      argdiffs::D,
                      new_args::Tuple,
                      proposal::Function,
                      proposal_args::Tuple) where {K <: AddressMap, D <: Diff}
    num_particles = length(ps)
    Threads.@threads for i in 1:num_particles
        _, p_cl, p_w = propose(proposal, ps.calls[i], proposal_args...)
        sel = selection(p_cl)
        merge!(sel, observations)
        _, ps.calls[i], u_w, _, _ = update(sel, ps.calls[i], argdiffs, new_args...)
        ps.lws[i] += u_w - p_w
    end
end

function filter_step!(observations::K,
                      params::P,
                      ps::Particles,
                      argdiffs::D,
                      new_args::Tuple,
                      proposal::Function,
                      proposal_args::Tuple) where {K <: AddressMap, P <: AddressMap, D <: Diff}
    num_particles = length(ps)
    Threads.@threads for i in 1:num_particles
        _, p_cl, p_w = propose(proposal, ps.calls[i], proposal_args...)
        sel = selection(p_cl)
        merge!(sel, observations)
        _, ps.calls[i], u_w, _, _ = update(sel, params, ps.calls[i], argdiffs, new_args...)
        ps.lws[i] += u_w - p_w
    end
end

function filter_step!(observations::K,
                      ps::Particles,
                      argdiffs::D,
                      new_args::Tuple,
                      pps::Ps,
                      proposal::Function,
                      proposal_args::Tuple) where {K <: AddressMap, Ps <: AddressMap, D <: Diff}
    num_particles = length(ps)
    Threads.@threads for i in 1:num_particles
        _, p_cl, p_w = propose(pps, proposal, ps.calls[i], proposal_args...)
        sel = selection(p_cl)
        merge!(sel, observations)
        _, ps.calls[i], u_w, _, _ = update(sel, params, ps.calls[i], argdiffs, new_args...)
        ps.lws[i] += u_w - p_w
    end
end

function filter_step!(observations::K,
                      params::P,
                      ps::Particles,
                      argdiffs::D,
                      new_args::Tuple,
                      pps::Ps,
                      proposal::Function,
                      proposal_args::Tuple) where {K <: AddressMap, P <: AddressMap, Ps <: AddressMap, D <: Diff}
    num_particles = length(ps)
    Threads.@threads for i in 1:num_particles
        _, p_cl, p_w = propose(pps, proposal, ps.calls[i], proposal_args...)
        sel = selection(p_cl)
        merge!(sel, observations)
        _, ps.calls[i], u_w, _, _ = update(sel, params, ps.calls[i], argdiffs, new_args...)
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

function resample!(ps::Particles)
    num_particles = length(ps.calls)
    ltw, lnw = nw(ps.lws)
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

function get_lmle(ps::Particles)
    return ps.lmle + lse(ps.lws) - log(length(ps))
end
