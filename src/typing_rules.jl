# ------------ Trace typing rules ------------ #


# Rules.
@abstract TracePrimitives trace(::Symbol, ::Normal) = Reals
@abstract TracePrimitives trace(::Symbol, ::Bernoulli) = Discrete{2}
@abstract TracePrimitives trace(::Symbol, jfn::JFunction{N, R, T}, args...) where {N, R, T} = get_trace_type(jfn.value)
