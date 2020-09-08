# ------------ Diff system ------------ #

# Define the algebra for propagation of diffs.
propagate(::NoChange, ::NoChange) = NoChange()
propagate(::UnknownChange, ::NoChange) = UnknownChange()
propagate(::NoChange, ::UnknownChange) = UnknownChange()
propagate(::UnknownChange, ::UnknownChange) = UnknownChange()
propagate(a::Type{NoChange}, v::Type{NoChange}) = NoChange()
propagate(a::Type{NoChange}, v::Type{UnknownChange}) = UnknownChange()
propagate(a::Type{UnknownChange}, v::Type{NoChange}) = UnknownChange()
propagate(a::Type{UnknownChange}, v::Type{UnknownChange}) = UnknownChange()
propagate(a::Type{K}, b::T) where {K, T} = propagate(K, T)

struct DiffPrimitives end

include("lib/numeric.jl")
include("lib/distributions.jl")
