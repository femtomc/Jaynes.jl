import Base.rand
rand(addr::T, d::Type, args) where T <: Union{Symbol, Pair{Symbol, Int64}} = rand(d(args...))
rand(addr::T, lit::K) where {T <: Union{Symbol, Pair{Symbol, Int64}}, 
                             K <: Union{Number, AbstractArray}} = lit

abstract type Trace end

abstract type RecordSite end

struct ChoiceSite{T} <: RecordSite
    val::T
    score::Float64
end

mutable struct CallSite{T <: Trace, J, K} <: RecordSite
    trace::T
    fn::Function
    args::J
    ret::K
end

mutable struct HierarchicalTrace <: Trace
    chm::Dict{Address, RecordSite}
    score::Float64
    HierarchicalTrace() = new(Dict{Address, ChoiceSite}(), 0.0)
end

mutable struct VectorizedTrace{T <: Trace} <: Trace
    sub::PersistentVector{T}
    score::Float64
    VectorizedTrace() = new{HierarchicalTrace}(PersistentHashMap{HierarchicalTrace}(), 0.0)
end

Trace() = HierarchicalTrace()

get_func(tr::Trace) = tr.func
get_args(tr::Trace) = tr.args
get_score(tr::Trace) = tr.score
get_chm(tr::Trace) = tr.chm
get_retval(tr::Trace) = tr.retval
