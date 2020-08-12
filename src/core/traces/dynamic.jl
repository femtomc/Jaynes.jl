const DynamicTrace = DynamicMap{Value}

struct DynamicCallSite{J, K} <: CallSite
    trace::DynamicTrace
    score::Float64
    fn::Function
    args::J
    ret::K
end

const DynamicDiscard = DynamicMap{Value}
