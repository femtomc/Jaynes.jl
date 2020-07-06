function metropolis_hastings(call::BlackBoxCallSite,
                             sel::UnconstrainedSelection)
    ctx = Regenerate(call.trace, sel)
    ret = ctx(call.fn, call.args...)
    log(rand()) < ctx.tr.score && return (BlackBoxCallSite(ctx.tr, call.fn, call.args, ret), true)
    return (call, false)
end

function metropolis_hastings(call::BlackBoxCallSite,
                             sel::UnconstrainedSelection,
                             obs::ConstrainedSelection)
    ctx = Regenerate(call.trace, sel, obs)
    prop, discard = ctx(call.fn, call.args...)
    log(rand()) < ctx.tr.score && return (BlackBoxCallSite(ctx.tr, call.fn, call.args, ret), true)
    return (call, false)
end

function metropolis_hastings(call::BlackBoxCallSite,
                             proposal::Function,
                             proposal_args::Tuple,
                             sel::UnconstrainedSelection)
    # Proposal.
    prop_ctx = Proposal(Trace())
    prop = prop_ctx(proposal, call, proposal_args...)
    
    # Update.
    update_ctx = Update(call.trace, selection(prop))
    update, discard = update_ctx(call.fn, call.args...)

    # Score.
    s_ctx = Score(discard)
    score = s_ctx(proposal, update, proposal_args...)

    # Accept/reject.
    ratio = update.score - prop.score + sc
    log(rand()) < ratio && return (BlackBoxCallSite(ctx.tr, call.fn, call.args, ret), true)
    return (call, false)
end
