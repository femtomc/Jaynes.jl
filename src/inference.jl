# Utility structure for collections of samples.
mutable struct Particles{C}
    calls::Vector{C}
    lws::Vector{Float64}
    lmle::Float64
end

include("inference/is.jl")
include("inference/pf.jl")
include("inference/mh.jl")
include("inference/vi.jl")
