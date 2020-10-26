const DynamicTrace = DynamicMap{Choice}

get_choices(dt::DynamicTrace) = dt.tree

struct DynamicCallSite{J, K} <: CallSite
    trace
    score::Float64
    fn
    args::J
    ret::K
end

@inline get_model(dcs::DynamicCallSite) = dcs.fn

@inline isempty(dcs::DynamicCallSite) = false

function projection(tr::DynamicTrace, tg::Target)
    weight, projected = 0.0, Trace()
    for (k, v) in shallow_iterator(tr)
        ss = get_sub(tg, k)
        w, sub = projection(v, ss)
        weight += w
        set_sub!(projected, k, sub)
    end
    weight, projected
end

@inline projection(cs::DynamicCallSite, tg::Empty) = 0.0, Empty()
@inline projection(cs::DynamicCallSite, tg::SelectAll) = cs.score, get_trace(cs)
@inline projection(cs::DynamicCallSite, tg::Target) = projection(cs.trace, tg)

@inline filter(fn, cs::DynamicCallSite) = filter(fn, cs.trace)
@inline filter(fn, addr, cs::DynamicCallSite) = filter(fn, addr, cs.trace)

@inline select(cs::DynamicCallSite) = select(cs.trace)

const DynamicDiscard = DynamicMap{Choice}
