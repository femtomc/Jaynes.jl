const VectorTrace = VectorMap{Choice}

struct VectorizedCallSite{F, A, R} <: CallSite
    trace::VectorTrace
    score::Float64
    fn::F
    len::Int
    args::A
    ret::Vector{R}
end

const VectorizedDiscard = DynamicMap{Choice}
