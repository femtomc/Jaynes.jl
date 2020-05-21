import Base.rand
rand(addr::T, d::Type, args) where T <: Union{Symbol, Pair} = rand(d(args...))

struct ChoiceOrCall{T}
    val::T
    lpdf::Float64
    d::Distribution
end

const Address = Union{Symbol, Pair}

mutable struct Trace
    chm::Dict{Address, ChoiceOrCall}
    obs::Dict{Address, Real}
    stack::Vector{Any}
    score::Float64
    Trace() = new(Dict{Symbol, Any}(), Dict{Symbol, Any}(), Core.TypeName[], 0.0)
end

function Trace(chm::Dict{T, K}) where {T <: Union{Symbol, Pair}, K <: Real}
    tr = Trace()
    tr.obs = chm
    return tr
end

# Merge observations and a choice map.
function merge(obs::Dict{Address, Real}, 
               chm::Dict{Address, ChoiceOrCall})
    obs_ks = collect(keys(obs))
    chm_ks = collect(keys(chm))
    map(chm_ks) do k
        k in obs_ks && error("SupportError: proposal has address on observed value.")
    end
    out = Dict{Address, Real}(map(chm_ks) do k
        k in obs_ks && return k => obs[k]
        return k => chm[k].val
    end)
    map(obs_ks) do k
        out[k] = obs[k]
    end
    return out
end

# Required to track nested calls in IR.
import Base: push!, pop!
function push!(tr::Trace, call::Symbol)
    isempty(tr.stack) && begin
        push!(tr.stack, call)
        return
    end
    push!(tr.stack, tr.stack[end] => call)
end

function pop!(tr::Trace)
    pop!(tr.stack)
end
