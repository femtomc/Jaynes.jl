@abstract DiffPrimitives (::D)(args...) where D <: Distributions.Distribution = propagate(args...)
@abstract DiffPrimitives (::Type{D})(args...) where D <: Distributions.Distribution = propagate(args...)

@abstract DiffPrimitives rand(::D) where D <: Distribution = Change
@abstract DiffPrimitives rand(::Symbol, args...) = propagate(args...)
@abstract DiffPrimitives rand(::Random._GLOBAL_RNG, ::D) where D <: Distribution = Change
@abstract DiffPrimitives rand(::Symbol, call::Function, args...) = propagate(args...)
