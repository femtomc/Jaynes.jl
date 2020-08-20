const DynamicTrace = DynamicMap{Choice}

get_choices(dt::DynamicTrace) = dt.tree

struct DynamicCallSite{J, K} <: CallSite
    trace::DynamicTrace
    score::Float64
    fn::Function
    args::J
    ret::K
end
@inline isempty(dcs::DynamicCallSite) = false

function projection(tr::DynamicTrace, tg::Target)
    weight = 0.0
    for (k, v) in shallow_iterator(tr)
        ss = get_sub(tg, k)
        weight += projection(v, ss)
    end
    weight
end

projection(cs::DynamicCallSite, tg::Empty) = 0.0
projection(cs::DynamicCallSite, tg::SelectAll) = cs.score
projection(cs::DynamicCallSite, tg::Target) = project(c.trace, tg)

filter(fn, cs::DynamicCallSite) = filter(fn, cs.trace)
filter(fn, addr, cs::DynamicCallSite) = filter(fn, addr, cs.trace)

const DynamicDiscard = DynamicMap{Choice}
