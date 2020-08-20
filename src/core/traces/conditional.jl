const ConditionalTrace = ConditionalMap{Choice}

get_choices(dt::ConditionalTrace) = (cond, branch)

struct ConditionalCallSite{J, K} <: CallSite
    trace::ConditionalTrace
    score::Float64
    fn::Function
    args::J
    ret::K
end
@inline isempty(dcs::ConditionalCallSite) = false

function projection(tr::ConditionalTrace, tg::Target)
    weight = 0.0
    for (k, v) in shallow_iterator(tr)
        ss = get_sub(tg, k)
        weight += projection(v, ss)
    end
    weight
end

projection(cs::ConditionalCallSite, tg::Empty) = 0.0
projection(cs::ConditionalCallSite, tg::SelectAll) = cs.score
projection(cs::ConditionalCallSite, tg::Target) = project(c.trace, tg)

filter(fn, cs::ConditionalCallSite) = filter(fn, cs.trace)
filter(fn, addr, cs::ConditionalCallSite) = filter(fn, addr, cs.trace)

const ConditionalDiscard = DynamicMap{Choice}
