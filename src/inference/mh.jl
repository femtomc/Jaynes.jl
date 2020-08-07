function metropolis_hastings(sel::K,
                             call::C) where {K <: UnconstrainedSelection, C <: CallSite}
    ret, cl, w, retdiff, d = regenerate(sel, call)
    log(rand()) < w && return (cl, true)
    return (call, false)
end

function metropolis_hastings(sel::K,
                             ps::P,
                             call::C) where {K <: UnconstrainedSelection, P <: Parameters, C <: CallSite}
    ret, cl, w, retdiff, d = regenerate(sel, ps, call)
    log(rand()) < w && return (cl, true)
    return (call, false)
end

function metropolis_hastings(call::C,
                             proposal::Function,
                             proposal_args::Tuple) where C <: CallSite
    p_ret, p_cl, p_w = propose(proposal, call, proposal_args...)
    s = selection(p_cl)
    u_ret, u_cl, u_w, retdiff, d = update(s, call.fn, NoChange(), call.args...)
    d_s = selection(d)
    s_ret, s_w = score(d_s, proposal, u_cl, proposal_args...)
    ratio = u_w - p_w + s_w
    log(rand()) < ratio && return (u_cl, true)
    return (call, false)
end

function metropolis_hastings(ps::P,
                             call::C,
                             proposal::Function,
                             proposal_args::Tuple) where {P <: Parameters, C <: CallSite}
    p_ret, p_cl, p_w = propose(proposal, call, proposal_args...)
    s = selection(p_cl)
    u_ret, u_cl, u_w, retdiff, d = update(s, ps, call.fn, NoChange(), call.args...)
    d_s = selection(d)
    s_ret, s_w = score(d_s, proposal, u_cl, proposal_args...)
    ratio = u_w - p_w + s_w
    log(rand()) < ratio && return (u_cl, true)
    return (call, false)
end

function metropolis_hastings(call::C,
                             pps::Ps,
                             proposal::Function,
                             proposal_args::Tuple) where {Ps <: Parameters, C <: CallSite}
    p_ret, p_cl, p_w = propose(pps, proposal, call, proposal_args...)
    s = selection(p_cl)
    u_ret, u_cl, u_w, retdiff, d = update(s, call.fn, NoChange(), call.args...)
    d_s = selection(d)
    s_ret, s_w = score(d_s, pps, proposal, u_cl, proposal_args...)
    ratio = u_w - p_w + s_w
    log(rand()) < ratio && return (u_cl, true)
    return (call, false)
end

function metropolis_hastings(ps::P,
                             call::C,
                             pps::Ps,
                             proposal::Function,
                             proposal_args::Tuple) where {P <: Parameters, Ps <: Parameters, C <: CallSite}
    p_ret, p_cl, p_w = propose(pps, proposal, call, proposal_args...)
    s = selection(p_cl)
    u_ret, u_cl, u_w, retdiff, d = update(s, ps, call.fn, NoChange(), call.args...)
    d_s = selection(d)
    s_ret, s_w = score(d_s, pps, proposal, u_cl, proposal_args...)
    ratio = u_w - p_w + s_w
    log(rand()) < ratio && return (u_cl, true)
    return (call, false)
end
