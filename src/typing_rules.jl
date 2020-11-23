# ------------ Trace typing rules ------------ #

struct TraceTypingInterpreter <: InterpretationContext end

# Fallback.
absint(ctx::TraceTypingInterpreter, args...) = Union{}

# Distributions.
absint(ctx::TraceTypingInterpreter, x::D, args...) where D <: Distributions.Distribution =  x
function absint(ctx::TraceTypingInterpreter, ::Type{D}, args...) where D <: Distributions.Distribution
    all(map(args) do a
            _lit(a)
        end) || return D
    return D(args...)
end

# Learnables.
@abstract TraceTypingInterpreter learnable(::Symbol) = Any

# Calls to trace.
@abstract TraceTypingInterpreter trace(::Symbol, ::Type{DiscreteUniform}, args...) = Integers{1}
@abstract TraceTypingInterpreter trace(::Symbol, d::DiscreteUniform, args...) = DiscreteInterval{d.a, d.b}
@abstract TraceTypingInterpreter trace(::Symbol, ::Type{Uniform}, args...) = Reals{1}
@abstract TraceTypingInterpreter trace(::Symbol, d::Uniform, args...) = RealInterval(d.a, d.b)
@abstract TraceTypingInterpreter trace(::Symbol, ::Type{Gamma}, args...) = PositiveReals{1}
@abstract TraceTypingInterpreter trace(::Symbol, ::Gamma, args...) = PositiveReals{1}
@abstract TraceTypingInterpreter trace(::Symbol, ::Type{Normal}, args...) = Reals{1}
@abstract TraceTypingInterpreter trace(::Symbol, ::Normal, args...) = Reals{1}
@abstract TraceTypingInterpreter trace(::Symbol, ::Type{Bernoulli}, args...) = Discrete{2}
@abstract TraceTypingInterpreter trace(::Symbol, ::Bernoulli, args...) = Discrete{2}
@abstract TraceTypingInterpreter trace(::Symbol, jfn::JFunction{N, R, T}, args...) where {N, R, T} = get_trace_type(jfn)

# Combinators.
@abstract TraceTypingInterpreter function trace(::Symbol, u::Gen.Unfold, args...)
    tt = get_trace_type(u.kernel)
    List{tt}
end
@abstract TraceTypingInterpreter function trace(::Symbol, u::Gen.Map, args...)
    List{get_trace_type(u.value.kernel)}
end
