# ------------ Trace typing rules ------------ #

abst(args...) = Union{}

_lit(v::Variable) = false
_lit(v::Type) = false
_lit(v) = true

# Distributions.
abst(x::D, args...) where D <: Distributions.Distribution =  x
function abst(::Type{D}, args...) where D <: Distributions.Distribution
    all(map(args) do a
            _lit(a)
        end) || return D
    return D(args...)
end

# Learnables.
abst(::typeof(learnable), ::Symbol) = Any

# Calls to trace.
abst(::typeof(trace), ::Symbol, ::Type{DiscreteUniform}) = Integers{1}
abst(::typeof(trace), ::Symbol, d::DiscreteUniform) = DiscreteInterval{d.a, d.b}
abst(::typeof(trace), ::Symbol, ::Type{Uniform}) = Reals{1}
abst(::typeof(trace), ::Symbol, d::Uniform) = RealInterval(d.a, d.b)
abst(::typeof(trace), ::Symbol, ::Type{Gamma}) = PositiveReals{1}
abst(::typeof(trace), ::Symbol, ::Gamma) = PositiveReals{1}
abst(::typeof(trace), ::Symbol, ::Type{Normal}) = Reals{1}
abst(::typeof(trace), ::Symbol, ::Normal) = Reals{1}
abst(::typeof(trace), ::Symbol, ::Type{Bernoulli}) = Discrete{2}
abst(::typeof(trace), ::Symbol, ::Bernoulli) = Discrete{2}
abst(::typeof(trace), ::Symbol, jfn::JFunction{N, R, T}, args...) where {N, R, T} = get_trace_type(jfn)

# Combinators.
function abst(::typeof(trace), ::Symbol, u::Gen.Unfold, args...)
    tt = get_trace_type(u.kernel)
    List{tt}
end
function abst(::typeof(trace), ::Symbol, u::Gen.Map, args...)
    List{get_trace_type(u.value.kernel)}
end
