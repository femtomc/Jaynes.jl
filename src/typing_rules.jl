# ------------ Trace typing rules ------------ #


# Rules.
@abstract TracePrimitives (::D)(args...) where D <: Distributions.Distribution =  D
@abstract TracePrimitives (::Type{D})(args...) where D <: Distributions.Distribution = D
@abstract TracePrimitives trace(::Symbol, ::Normal) = Reals
@abstract TracePrimitives trace(::Symbol, ::Bernoulli) = Discrete{2}
@abstract TracePrimitives trace(::Symbol, jfn::JFunction{N, R, T}, args...) where {N, R, T} = get_trace_type(jfn.value)
@abstract TracePrimitives function trace(::Symbol, u::Gen.Unfold, args...) where {N, R, T}
    val = u.value
    List{get_trace_type(val.kernel)}
end
