mutable struct NVector{T,S} <: AbstractVector{T}
    data::NTuple{S, T}
end
Base.size(xs::NVector) = (length(xs.data),)
Base.getindex(xs::NVector, i::Integer) = xs.data[i]
Base.setindex!(xs::NVector, v, i::Integer)  = xs.data = ntuple(j -> j == i ? v : xs.data[j], length(xs.data))

# Maps.
include("core/address_map.jl")

# Selections.
include("core/selections.jl")

# Visitor.
include("core/visitor.jl")

# Traces.
include("core/traces.jl")
Trace() = DynamicTrace()

# Learnables and gradients.
include("core/learnables.jl")
