const VectorTrace = VectorMap{Choice}

get_choices(vt::VectorTrace) = vt.vector

struct VectorCallSite{T, F, A, R} <: CallSite
    trace::VectorTrace
    score::Float64
    fn::F
    len::Int
    args::A
    ret::Vector{R}
    VectorCallSite{T}(tr, sc, fn::F, len, args::A, ret::Vector{R}) where {T, F, A, R} = new{T, F, A, R}(tr, sc, fn, len, args, ret)
end
@inline isempty(vcs::VectorCallSite) = false

function projection(tr::VectorTrace, tg::Target)
    weight = 0.0
    for (k, v) in shallow_iterator(tr)
        ss = get_sub(tg, k)
        weight += projection(v, ss)
    end
    weight
end

projection(cs::VectorCallSite, tg::Empty) = 0.0
projection(cs::VectorCallSite, tg::SelectAll) = cs.score
projection(cs::VectorCallSite, tg::Target) = project(c.trace, tg)

const VectorDiscard = DynamicMap{Choice}
