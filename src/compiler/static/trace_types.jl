# ------------ Trace typing system ------------ #

# Utilities.
keys(::Type{NamedTuple{K, V}}) where {K, V} = K
keys(::NamedTuple{K, V}) where {K, V} = K
value_types(::Type{NamedTuple{K, V}}) where {K, V} = V
NamedTuple{K, V}() where {K, V} = NamedTuple{K}(map(V.parameters) do p
                                                    p()
                                                end)

function Base.:(<<)(nt1::Type{NamedTuple{K1, V1}}, nt2::Type{NamedTuple{K2, V2}}) where {K1, K2, V1, V2}
    for k in K1
        k in K2 || return false
    end
    for (v, k) in zip(V1.parameters, V2.parameters)
        v << k || return false
    end
    true
end

# Context.
struct TracePrimitives end

# ------------ Support types ------------ #

abstract type SupportType end

abstract type BaseLebesgue <: SupportType end
pretty(::Type{BaseLebesgue}) = :lebesgue

abstract type BaseCounting <: SupportType end
pretty(::Type{BaseCounting}) = :counting

struct Reals{N} <: BaseLebesgue end
Base.:(<<)(::Type{Reals{N}}, ::Type{Reals{N}}) where N = true
Base.:(<<)(::Type{Reals}, a) = false

struct PositiveReals{N} <: BaseLebesgue end
Base.:(<<)(::Type{PositiveReals{N}}, ::Type{PositiveReals{N}}) where N = true
Base.:(<<)(::Type{PositiveReals}, a) = false

struct Integers <: BaseCounting end
Base.:(<<)(::Type{Integers}, ::Type{Reals}) = true
Base.:(<<)(::Type{Integers}, ::Type{Integers}) = true
Base.:(<<)(::Type{Integers}, a) = false

struct Discrete{N} <: BaseCounting end

struct List{N} <: SupportType
    tt::N
end
List{N}() where N = List{N}(N())

# Inferred record type is just a NamedTuple.
const TraceType = NamedTuple

# ------------ Tracer ------------ #

# See Mjolnir `trace` for original source.
function trace_with_partial_cleanup(P, Ts...)
    tr = Mjolnir.Trace(P)
    try
        argnames = [argument!(tr.ir, T) for T in Ts]
        for (T, x) in zip(Ts, argnames)
            T isa Union{Mjolnir.Partial, Mjolnir.Shape} && Mjolnir.node!(tr, T, x)
        end
        args = [T isa Const ? T.value : arg for (T, arg) in zip(Ts, argnames)]
        args, Ts = Mjolnir.replacement(P, args, Ts)
        if (T = Mjolnir.partial(tr.primitives, Ts...)) != nothing
            tr.total += 1
            Mjolnir.return!(tr.ir, push!(tr.ir, stmt(Expr(:call, args...), type = T)))
        else
            Mjolnir.return!(tr.ir, tracecall!(tr, args, Ts...))
        end
        # @info "$(tr.total) functions traced."
        return partial_cleanup!(tr.ir)
    catch e
        throw(Mjolnir.TraceError(e, tr.stack))
    end
end

# Create new defaults - tracer will try TracePrimitives, then fallback to Defaults.
TraceDefaults() = Multi(TracePrimitives(), Mjolnir.Defaults())

# Type inference.
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

# Extract trace type from inferred IR.
function trace_type(tr)
    ks = Set{Symbol}([])
    types = []
    for (v, st) in tr
        st.expr isa Expr || continue
        st.expr.head == :call || continue
        st.expr.args[1] == trace || continue
        st.expr.args[2] isa QuoteNode || continue
        push!(ks, st.expr.args[2].value)
        if st.type isa Type && st.type <: SupportType
            push!(types, st.type())
        elseif st.type <: NamedTuple
            new = NamedTuple{keys(st.type)}(map(value_types(st.type).parameters) do p
                                                       p()
                                                   end)
            push!(types, new)
        end
    end
    NamedTuple{tuple(ks...)}(types)
end
