function metropolis_hastings(call::CallSite,
                             sel::UnconstrainedSelection)
    args = call.args
    ctx = Regenerate(call.trace, sel)
    prop, discard = trace(ctx, call.fn, call.args)
    log(rand()) < prop.trace.score && return (prop, true)
    return (call, false)
end

function metropolis_hastings(call::CallSite,
                             sel::UnconstrainedSelection,
                             obs::ConstrainedSelection)
    args = call.args
    ctx = Regenerate(call.trace, sel, obs)
    prop, discard = trace(ctx, call.fn, call.args)
    log(rand()) < prop.trace.score && return (prop, true)
    return (call, false)
end

function metropolis_hastings(call::CallSite,
                             proposal::Function,
                             proposal_args::Tuple,
                             sel::UnconstrainedSelection)
    # Proposal.
    prop_ctx = Proposal(Trace())
    prop = trace(prop_ctx, proposal, call, proposal_args...)
    
    # Update.
    update_ctx = Update(call.trace, selection(prop))
    update, discard = trace(update_ctx, call.fn, call.args...)

    # Score.
    s_ctx = Score(discard)
    score = trace(s_ctx, proposal, update, proposal_args...)

    # Accept/reject.
    ratio = update.score - prop.score + sc
    log(rand()) < ratio && return (prop, true)
    return (call, false)
end
