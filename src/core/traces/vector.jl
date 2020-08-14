const VectorTrace = VectorMap{Choice}

struct VectorCallSite{F, A, R} <: CallSite
    trace::VectorTrace
    score::Float64
    fn::F
    len::Int
    args::A
    ret::Vector{R}
end

const VectorDiscard = DynamicMap{Choice}
