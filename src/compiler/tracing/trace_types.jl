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

# ------------ Support types ------------ #

abstract type SupportType end

abstract type BaseLebesgue <: SupportType end
pretty(::Type{BaseLebesgue}) = :lebesgue

abstract type BaseCounting <: SupportType end
pretty(::Type{BaseCounting}) = :counting

struct Reals{N} <: BaseLebesgue end
Base.:(<<)(::Type{Reals{N}}, ::Type{Reals{N}}) where N = true
Base.:(<<)(::Type{Reals}, a) = false

struct RealInterval <: BaseLebesgue
    a
    b
end

struct DiscreteInterval{A, B} <: BaseCounting end

struct PositiveReals{N} <: BaseLebesgue end
Base.:(<<)(::Type{PositiveReals{N}}, ::Type{PositiveReals{N}}) where N = true
Base.:(<<)(::Type{PositiveReals}, a) = false

struct Integers{N} <: BaseCounting end
Base.:(<<)(::Type{Integers{N}}, ::Type{Reals{N}}) where N = true
Base.:(<<)(::Type{Integers{N}}, ::Type{Integers{N}}) where N = true
Base.:(<<)(::Type{Integers}, a) = false

struct Discrete{N} <: BaseCounting end

struct List{N} <: SupportType
    tt::N
end
List{N}() where N = List{N}(N())

# Inferred record type is just a NamedTuple.
const TraceType = NamedTuple

# ------------ Tracer ------------ #

# Type inference.
function infer_support_types(fn, arg_types...)
    ir = lower_to_ir(fn, arg_types...)
    dynamic_address_check(ir) && return missing
    try
        tr = ir |> prepare_ir! |> infer!
        tr
    catch e
        @info "Failed to trace $(fn).\nCause: $e\n\nReverting to untyped IR."
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
        if !(st.type isa Type)
            push!(types, st.type)
        elseif st.type isa Type && st.type <: SupportType
            push!(types, st.type())
        elseif lift(st.type) <: NamedTuple
            new = NamedTuple{keys(st.type)}(map(value_types(st.type).parameters) do p
                                                p()
                                            end)
            push!(types, new)
        end
    end
    NamedTuple{tuple(ks...)}(types)
end
