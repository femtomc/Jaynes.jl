function metropolis_hastings(call::GenericCallSite,
                             sel::UnconstrainedSelection)
    ret, cl, w, retdiff, d = regenerate(sel, call, call.args...)
    log(rand()) < w && return (cl, true)
    return (call, false)
end

function metropolis_hastings(call::GenericCallSite,
                             proposal::Function,
                             proposal_args::Tuple,
                             sel::UnconstrainedSelection)
    p_ret, p_cl, p_w= propose(proposal, call, proposal_args...)
    s = selection(p_cl)
    u_ret, u_cl, u_w, retdiff, d = update(s, call.fn, call.args...)
    d_s = selection(d)
    s_ret, s_w = score(d_s, proposal, u_cl, proposal_args...)
    ratio = u_w - p_w + s_w
    log(rand()) < ratio && return (u_cl, true)
    return (call, false)
end

# ------------ Documentation ------------ #

@doc(
"""
```julia
call, accepted, metropolis_hastings(call::GenericCallSite,
                                    sel::UnconstrainedSelection)
```

Perform a Metropolis-Hastings step by proposing new choices using the prior at addressed specified by `sel`. Returns a call site, as well as a Boolean value `accepted` to indicate if the proposal was accepted or rejected.

```julia
call, accepted = metropolis_hastings(call::GenericCallSite,
                                     proposal::Function,
                                     proposal_args::Tuple,
                                     sel::UnconstrainedSelection)
```

Perform a Metropolis-Hastings step by proposing new choices using a custom proposal at addressed specified by `sel`. Returns a call site, as well as a Boolean value `accepted` to indicate if the proposal was accepted or rejected.
""", metropolis_hastings)
