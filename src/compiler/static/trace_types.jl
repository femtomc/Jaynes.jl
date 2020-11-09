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

# Inferred record type is just a NamedTuple.
const TraceType = NamedTuple

function infer_support_types(fn, arg_type...)
    ir = lower_to_ir(fn, arg_type...)
    dynamic_address_check(ir) && return missing
    try
        tr = trace_with_partial_cleanup(TraceDefaults(), typeof(fn), arg_type...)
    catch e
        @info "Failed to trace $(fn). Cause: $e\n\nReverting to untyped IR."
        return missing
    end
end

function trace_type(tr)
    keys = Set{Symbol}([])
    types = []
    for (v, st) in tr
        st.expr isa Expr || continue
        st.expr.head == :call || continue
        st.expr.args[1] == trace || continue
        st.expr.args[2] isa QuoteNode || continue
        push!(keys, st.expr.args[2].value)
        if st.type isa Type && st.type <: SupportType
            push!(types, st.type())
        else
            push!(types, st.type)
        end
    end
    NamedTuple{tuple(keys...)}(types)
end
