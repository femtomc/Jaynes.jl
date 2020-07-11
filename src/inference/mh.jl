function metropolis_hastings(call::BlackBoxCallSite,
                             sel::UnconstrainedSelection)
    ctx = Regenerate(call.trace, sel)
    ret = ctx(call.fn, call.args...)
    log(rand()) < ctx.weight && return (BlackBoxCallSite(ctx.tr, call.fn, call.args, ret), true)
    return (call, false)
end

function metropolis_hastings(call::BlackBoxCallSite,
                             sel::UnconstrainedSelection,
                             obs::ConstrainedSelection)
    ctx = Regenerate(call.trace, sel, obs)
    prop, discard = ctx(call.fn, call.args...)
    log(rand()) < ctx.weight && return (BlackBoxCallSite(ctx.tr, call.fn, call.args, ret), true)
    return (call, false)
end

function metropolis_hastings(call::BlackBoxCallSite,
                             proposal::Function,
                             proposal_args::Tuple,
                             sel::UnconstrainedSelection)
    # Proposal.
    prop_ctx = Proposal(Trace())
    prop, p_weight = prop_ctx(proposal, call, proposal_args...)
    
    # Update.
    update_ctx = Update(call.trace, selection(prop))
    update_cs, u_weight, retdiff, discard = update_ctx(call.fn, call.args...)

    # Score.
    s_ctx = Score(discard)
    s_weight = s_ctx(proposal, update_cs, proposal_args...)

    # Accept/reject.
    ratio = u_weight - p_weight + s_weight
    log(rand()) < ratio && return (BlackBoxCallSite(ctx.tr, call.fn, call.args, ret), true)
    return (call, false)
end
