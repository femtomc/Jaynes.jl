function metropolis_hastings(call::CallSite,
                             sel::UnconstrainedSelection)
    args = call.args
    ctx = Regenerate(call.trace, sel)
    prop, discard = trace(ctx, call.fn, call.args)
    log(rand()) < prop.trace.score && return (prop, true)
    return (trace, false)
end

function metropolis_hastings(call::CallSite,
                             sel::UnconstrainedSelection,
                             obs::ConstrainedSelection)
    args = call.args
    ctx = Regenerate(call.trace, sel, obs)
    prop, discard = trace(ctx, call.fn, call.args)
    log(rand()) < prop.trace.score && return (prop, true)
    return (trace, false)
end

# TODO: custom proposals.
function metropolis_hastings(call::CallSite,
                             proposal::Function,
                             proposal_args::Tuple,
                             sel::UnconstrainedSelection)
    args = call.args
    ctx = Regenerate(call.trace, sel)
    prop, weight = trace(ctx, call.fn, call.args)
    log(rand()) < weight && return (prop, true)
    return (trace, false)
end

function metropolis_hastings(call::CallSite,
                             proposal::Function,
                             proposal_args::Tuple,
                             sel::UnconstrainedSelection,
                             obs::ConstrainedSelection)
    args = call.args
    ctx = Regenerate(call.trace, sel, obs)
    prop, weight = trace(ctx, call.fn, call.args)
    log(rand()) < weight && return (prop, true)
    return (trace, false)
end
