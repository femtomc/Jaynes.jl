import Base.rand
rand(addr::T, d::Type, args) where T <: Union{Symbol, Pair} = rand(d(args...))

struct ChoiceOrCall
    val::Union{Int64, Float64}
    score::Float64
end

const Address = Union{Symbol, Pair}

mutable struct Trace
    chm::Dict{Address, ChoiceOrCall}
    obs::Dict{Address, Union{Int64, Float64}}
    stack::Vector{Union{Pair, Symbol}}
    score::Float64
    func::Function
    args::Tuple
    retval::Any
    Trace() = new(Dict{Address, ChoiceOrCall}(), 
                  Dict{Address, Union{Int64, Float64}}(), 
                  Symbol[], 
                  0.0)
end

function Trace(obs::Dict{Address, Union{Int64, Float64}})
    tr = Trace()
    tr.obs = obs
    return tr
end

get_func(tr::Trace) = tr.func
get_args(tr::Trace) = tr.args
get_score(tr::Trace) = tr.score
get_chm(tr::Trace) = tr.chm
get_retval(tr::Trace) = tr.retval

# Merge observations and a choice map.
function merge(obs::Dict{Address, Union{Int64, Float64}}, 
               chm::Dict{Address, ChoiceOrCall})
    obs_ks = collect(keys(obs))
    chm_ks = collect(keys(chm))
    map(chm_ks) do k
        k in obs_ks && error("SupportError: proposal has address on observed value.")
    end
    out = Dict{Address, Union{Int64, Float64}}(map(chm_ks) do k
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
