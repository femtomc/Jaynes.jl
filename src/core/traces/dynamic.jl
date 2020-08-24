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

@inline projection(cs::DynamicCallSite, tg::Empty) = 0.0
@inline projection(cs::DynamicCallSite, tg::SelectAll) = cs.score
@inline projection(cs::DynamicCallSite, tg::Target) = projection(c.trace, tg)

@inline filter(fn, cs::DynamicCallSite) = filter(fn, cs.trace)
@inline filter(fn, addr, cs::DynamicCallSite) = filter(fn, addr, cs.trace)

@inline select(cs::DynamicCallSite) = select(cs.trace)

const DynamicDiscard = DynamicMap{Choice}
