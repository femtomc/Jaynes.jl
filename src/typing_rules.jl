# ------------ Trace typing rules ------------ #

abst(args...) = Union{}

# Distributions.
abst(::D, args...) where D <: Distributions.Distribution =  D
abst(::Type{D}, args...) where D <: Distributions.Distribution = D

# Learnables.
abst(::typeof(learnable), ::Symbol) = Any

# Calls to trace.
abst(::typeof(trace), ::Symbol, ::Type{DiscreteUniform}) = Integers{1}
abst(::typeof(trace), ::Symbol, ::Type{Uniform}) = Reals{1}
abst(::typeof(trace), ::Symbol, ::Type{Gamma}) = PositiveReals{1}
abst(::typeof(trace), ::Symbol, ::Type{Normal}) = Reals{1}
abst(::typeof(trace), ::Symbol, ::Type{Bernoulli}) = Discrete{2}
abst(::typeof(trace), ::Symbol, jfn::JFunction{N, R, T}, args...) where {N, R, T} = get_trace_type(jfn)

# Combinators.
function abst(::typeof(trace), ::Symbol, u::Gen.Unfold, args...)
    tt = get_trace_type(u.kernel)
    List{tt}
end
function abst(::typeof(trace), ::Symbol, u::Gen.Map, args...)
    List{get_trace_type(u.value.kernel)}
end
