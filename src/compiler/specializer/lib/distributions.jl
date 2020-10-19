# ------------ Runtime ------------ #

(d::Type{D})(args...) where {D <: Distribution, Df <: Diffed} = d(map(a -> Jaynes.unwrap(a), args)...)

# ------------ Abstract ------------ #

@abstract DiffPrimitives (::Distribution)(args...) = propagate(args...)
@abstract DiffPrimitives (::Type{D})(args...) where D <: Distributions.Distribution = propagate(args...)

@abstract DiffPrimitives rand(::D) where D <: Distribution = Change
@abstract DiffPrimitives rand(::Symbol, args...) = propagate(args...)
@abstract DiffPrimitives rand(::Random._GLOBAL_RNG, ::D) where D <: Distribution = Change
@abstract DiffPrimitives rand(::Symbol, call::Function, args...) = propagate(args...)

@abstract DiffPrimitives trace(::D) where D <: Distribution = Change
@abstract DiffPrimitives trace(::Symbol, args...) = propagate(args...)
@abstract DiffPrimitives trace(::Random._GLOBAL_RNG, ::D) where D <: Distribution = Change
@abstract DiffPrimitives trace(::Symbol, call::Function, args...) = propagate(args...)
