function metropolis_hastings(call::BlackBoxCallSite,
                             sel::UnconstrainedSelection)
    ctx = Regenerate(call.trace, sel)
    ret = ctx(call.fn, call.args...)
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

# ------------ Documentation ------------ #

@doc(
"""
```julia
call, accepted, metropolis_hastings(call::BlackBoxCallSite,
                                    sel::UnconstrainedSelection)
```

Perform a Metropolis-Hastings step by proposing new choices using the prior at addressed specified by `sel`. Returns a call site, as well as a Boolean value `accepted` to indicate if the proposal was accepted or rejected.

```julia
call, accepted = metropolis_hastings(call::BlackBoxCallSite,
                                     proposal::Function,
                                     proposal_args::Tuple,
                                     sel::UnconstrainedSelection)
```

Perform a Metropolis-Hastings step by proposing new choices using a custom proposal at addressed specified by `sel`. Returns a call site, as well as a Boolean value `accepted` to indicate if the proposal was accepted or rejected.
""", metropolis_hastings)
