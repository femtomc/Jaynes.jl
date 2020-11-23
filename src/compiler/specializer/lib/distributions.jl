# ------------ Runtime ------------ #

(d::Type{D})(args::Diffed...) where D <: Distributions.Distribution = d(map(unwrap, args)...)
(d::Type{D})(arg1::Diffed, others...) where D <: Distributions.Distribution = d(unwrap(arg1), others...)

# ------------ Abstract ------------ #

@abstract DiffInterpreter (::Distribution)(args...) = propagate(args...)
@abstract DiffInterpreter (::Type{D})(args...) where D <: Distributions.Distribution = propagate(args...)

@abstract DiffInterpreter rand(::D) where D <: Distribution = Change
@abstract DiffInterpreter rand(::Symbol, args...) = propagate(args...)
@abstract DiffInterpreter rand(::Random._GLOBAL_RNG, ::D) where D <: Distribution = Change
@abstract DiffInterpreter rand(::Symbol, call::Function, args...) = propagate(args...)

@abstract DiffInterpreter trace(::D) where D <: Distribution = Change
@abstract DiffInterpreter trace(::Symbol, args...) = consume(args...)
@abstract DiffInterpreter trace(::Random._GLOBAL_RNG, ::D) where D <: Distribution = Change
@abstract DiffInterpreter trace(::Symbol, call::Function, args...) = consume(args...)
@abstract DiffInterpreter trace(args...) = consume(args...)
