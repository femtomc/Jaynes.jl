import Base.rand
rand(addr::T, d::Type, args) where T <: Address = rand(d(args...))
rand(addr::T, lit::K) where {T <: Address, K <: Union{Number, AbstractArray}} = lit

abstract type RecordSite end

struct Choice{T} <: RecordSite
    val::T
    score::Float64
end

struct Call{T <: Trace} <: RecordSite
    subtrace::T
    score::Float64
end

abstract type Trace end

mutable struct HierarchicalTrace <: Trace
    chm::Dict{Address, RecordSite}
    score::Float64
    HierarchicalTrace() = new(Dict{Address, Choice}(), 0.0)
end

mutable struct VectorizedTrace{T <: Trace} <: Trace
    sub::PersistentVector{T}
    score::Float64
    VectorizedTrace() = new(PersistentHashMap{HierarchicalTrace}(), 0.0)
end

get_func(tr::Trace) = tr.func
get_args(tr::Trace) = tr.args
get_score(tr::Trace) = tr.score
get_chm(tr::Trace) = tr.chm
get_retval(tr::Trace) = tr.retval
