# ------------ Trace typing rules ------------ #

# Distributions.
@abstract TracePrimitives (::D)(args...) where D <: Distributions.Distribution =  D
@abstract TracePrimitives (::Type{D})(args...) where D <: Distributions.Distribution = D

# Calls to trace.
@abstract TracePrimitives trace(::Symbol, ::Normal) = Reals{1}
@abstract TracePrimitives trace(::Symbol, ::Bernoulli) = Discrete{2}
@abstract TracePrimitives trace(::Symbol, jfn::JFunction{N, R, T}, args...) where {N, R, T} = get_trace_type(jfn.value)

# Combinators.
@abstract TracePrimitives function trace(::Symbol, u::Gen.Unfold, args...)
    tt = get_trace_type(u.value.kernel)
    List{tt}
end
@abstract TracePrimitives trace(::Symbol, u::Gen.Map, args...) = List{get_trace_type(u.value.kernel)}
