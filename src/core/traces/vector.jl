const VectorTrace = VectorMap{Choice}

get_choices(vt::VectorTrace) = vt.vector

struct VectorCallSite{T, F, A, R} <: CallSite
    trace::VectorTrace
    score::Float64
    fn::F
    args::A
    ret::Vector{R}
    len::Int
    VectorCallSite{T}(tr, sc, fn::F, args::A, ret::Vector{R}, len) where {T, F, A, R} = new{T, F, A, R}(tr, sc, fn, args, ret, len)
end
@inline isempty(vcs::VectorCallSite) = false

function projection(tr::VectorTrace, tg::Target)
    weight, projected = 0.0, Trace()
    for (k, v) in shallow_iterator(tr)
        ss = get_sub(tg, k)
        w, sub = projection(v, ss)
        weight += w
        set_sub!(projected, k, sub)
    end
    weight, projected
end

projection(cs::VectorCallSite, tg::Empty) = 0.0
projection(cs::VectorCallSite, tg::SelectAll) = cs.score
projection(cs::VectorCallSite, tg::Target) = projection(cs.trace, tg)

filter(fn, cs::VectorCallSite) = filter(fn, cs.trace)
filter(fn, addr, cs::VectorCallSite) = filter(fn, addr, cs.trace)

const VectorDiscard = DynamicMap{Choice}
