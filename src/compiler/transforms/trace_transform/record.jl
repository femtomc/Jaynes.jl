# Note - currently copied from Gen.jl

struct ChoiceRecord{T, K}
    val::T
    score::Float64
    dist::K
end
