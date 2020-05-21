import Base.rand
rand(addr::T, d::Distribution) where T <: Union{Symbol, Pair} = rand(d)

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
function merge(obs::Dict{Address, Real}, chm::Dict{Address, ChoiceOrCall})
    obs_ks = keys(obs)
    chm_ks = collect(keys(chm))
    setd = collect(setdiff(obs_ks, chm_ks))
    out = Dict{Address, Real}(map(chm_ks) do k
        k in obs_ks && return k => obs[k]
        return k => chm[k].val
    end)
    map(setd) do k
        out[k] = obs[k]
    end
    return out
end

# Required to track nested calls in IR.
import Base: push!, pop!
function push!(tr::Trace, call::Symbol)
    call == :rand && begin
        push!(tr.stack, tr.stack[end])
        return
    end
    isempty(tr.stack) && begin
        push!(tr.stack, call)
        return
    end
    push!(tr.stack, tr.stack[end] => call)
end

function pop!(tr::Trace)
    pop!(tr.stack)
end

# Convenience macro.
macro trace(call)
    expr = quote
        tr -> begin
            tr() do
                $call
            end
            tr
        end
    end
    expr
end
