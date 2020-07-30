function metropolis_hastings(sel::K,
                             call::C) where {K <: UnconstrainedSelection, C <: CallSite}
    ret, cl, w, retdiff, d = regenerate(sel, call)
    log(rand()) < w && return (cl, true)
    return (call, false)
end

function metropolis_hastings(sel::K,
                             call::C,
                             proposal::Function,
                             proposal_args::Tuple) where {K <: UnconstrainedSelection, C <: CallSite}
    p_ret, p_cl, p_w = propose(proposal, call, proposal_args...)
    s = selection(p_cl)
    u_ret, u_cl, u_w, retdiff, d = update(s, call.fn, NoChange(), call.args...)
    d_s = selection(d)
    s_ret, s_w = score(d_s, proposal, u_cl, proposal_args...)
    ratio = u_w - p_w + s_w
    log(rand()) < ratio && return (u_cl, true)
    return (call, false)
end

function metropolis_hastings!(sel::K,
                              ps::Particles) where K <: UnconstrainedSelection
    num_particles = length(ps)
    Threads.@threads for i in 1 : num_particles
        old_w = ps.lws[i]
        old_score = get_score(ps.calls[i])
        ps.calls[i], _ = metropolis_hastings(sel, ps.calls[i])
        ps.lws[i] = old_w + get_score(ps.calls[i]) - old_score
    end
end

function metropolis_hastings!(sel::K,
                              ps::Particles,
                              proposal::Function,
                              proposal_args::Tuple) where K <: UnconstrainedSelection
    num_particles = length(ps)
    Threads.@threads for i in 1 : num_particles
        old_w = ps.lws[i]
        old_score = get_score(ps.calls[i])
        ps.calls[i], _ = metropolis_hastings(sel, ps.calls[i], proposal, proposal_args)
        ps.lws[i] = old_w + get_score(ps.calls[i]) - old_score
    end
end
