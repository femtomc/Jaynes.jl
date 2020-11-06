struct TracePrimitives end

# Support types.
abstract type SupportType end

abstract type BaseLebesgue <: SupportType end
pretty(::Type{BaseLebesgue}) = :lebesgue

abstract type BaseCounting <: SupportType end
pretty(::Type{BaseCounting}) = :counting

struct Reals <: BaseLebesgue end
struct PositiveReals <: BaseLebesgue end

struct Integers <: BaseCounting end
struct Discrete{N} <: BaseCounting end

TraceDefaults() = Multi(TracePrimitives(), Mjolnir.Defaults())

@abstract TracePrimitives trace(::Symbol, ::Normal) = Reals
@abstract TracePrimitives trace(::Symbol, ::Bernoulli) = Discrete{2}

# Inferred record type is just a NamedTuple.
const TraceType = NamedTuple

function infer_support_types(fn_type, arg_type...)
    try
        Mjolnir.trace(TraceDefaults(), fn_type, arg_type...)
    catch e
        @info "Failed to trace $(fn_type). Cause: $e\n\nReverting to untyped IR."
        return missing
    end
end

function trace_type(tr)
    keys = Set{Symbol}([])
    types = Set{Jaynes.SupportType}([])
    for (v, st) in tr
        st.expr isa Expr || continue
        st.expr.head == :call || continue
        st.expr.args[1] == rand || continue
        st.expr.args[2] isa QuoteNode || continue
        push!(keys, st.expr.args[2].value)
        push!(types, st.type())
    end
    NamedTuple{tuple(keys...)}(types)
end

#function display(nt::NamedTuple)
#    for (k, v) in zip(keys(nt), values(nt))
#        println(k => v)
#    end
#end
