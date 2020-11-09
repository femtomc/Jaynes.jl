# ------------ Trace typing rules ------------ #

# Distributions.
@abstract TracePrimitives (::D)(args...) where D <: Distributions.Distribution =  D
@abstract TracePrimitives (::Type{D})(args...) where D <: Distributions.Distribution = D

# Calls to trace.
@abstract TracePrimitives trace(::Symbol, ::Normal) = Reals
@abstract TracePrimitives trace(::Symbol, ::Bernoulli) = Discrete{2}
@abstract TracePrimitives trace(::Symbol, jfn::JFunction{N, R, T}, args...) where {N, R, T} = get_trace_type(jfn.value)

# Combinators.
@abstract TracePrimitives trace(::Symbol, u::Gen.Unfold, args...) = List{get_trace_type(u.value.kernel)}
@abstract TracePrimitives trace(::Symbol, u::Gen.Map, args...) = List{get_trace_type(u.value.kernel)}
