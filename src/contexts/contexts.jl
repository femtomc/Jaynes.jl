# Driver.
@dynamo function (mx::ExecutionContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    recurse!(ir)
    return ir
end

# Fix for _apply_iterate.
function f_push!(arr::Array, t::Tuple{}) end
f_push!(arr::Array, t::Array) = append!(arr, t)
f_push!(arr::Array, t::Tuple) = append!(arr, t)
f_push!(arr, t) = push!(arr, t)
function flatten(t::Tuple)
    arr = Any[]
    for sub in t
        f_push!(arr, sub)
    end
    return arr
end

function (mx::ExecutionContext)(::typeof(Core._apply_iterate), f, c::typeof(rand), args...)
    return mx(c, flatten(args)...)
end

include("generate.jl")
include("propose.jl")
include("score.jl")

# Vectorized utilities.
function keyset(sel::L, new_length::Int) where L <: ConstrainedSelection
    keyset = Set{Int}()
    for (k, _) in sel
        k > 0 && k <= n_len && push!(keyset, k)
    end
    return keyset
end

function keyset(sel::L, new_length::Int) where L <: UnconstrainedSelection
    keyset = Set{Int}()
    for (k, _) in sel
        k > 0 && k <= n_len && push!(keyset, k)
    end
    return keyset
end

function update_subtraces_and_ret!(sbtrs::Vector{T}, ret::Vector{K}, o_len::Int, n_len::Int) where {T <: Trace, K}
    for i in n_len + 1 : o_len
        pop!(subtraces)
        pop!(ret)
    end
end

function update(vcs::VectorizedCallSite{T, J, K}, sel::L, args::Vector) where {T, J, K, L <: ConstrainedSelection}
    n_len, o_len = length(args), length(vsc.args)
    ks = keyset(sel, n_len)
    score_acc = 0.0
    discard = ConstrainedHierarchicalSelection()
    for k in n_len + 1 : o_len
        sub = vcs.subtraces[k]
        score_acc += score(sub)
        discard[k] = selection(sub)
    end
    return discard, score_acc
end

function regenerate(vcs::VectorizedCallSite{T, J, K}, sel::L, args::Vector) where {T, J, K, L <: ConstrainedSelection}
    n_len, o_len = length(args), length(vsc.args)
    ks = keyset(sel, n_len)
    score_acc = 0.0
    for k in n_len + 1 : o_len
        sub = vcs.subtraces[k]
        score_acc += score(sub)
    end
    return score_acc
end

include("update.jl")
include("regenerate.jl")
