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

function discard_and_score_adj(vcs::VectorizedCallSite{T, J, K}, sel::L, args::Vector) where {T, J, K, L <: ConstrainedSelection}
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

function score_adj(vcs::VectorizedCallSite{T, J, K}, sel::L, args::Vector) where {T, J, K, L <: ConstrainedSelection}
    n_len, o_len = length(args), length(vsc.args)
    ks = keyset(sel, n_len)
    score_acc = 0.0
    for k in n_len + 1 : o_len
        sub = vcs.subtraces[k]
        score_acc += score(sub)
    end
    return score_acc
end

function retrace_retained!(vcs::VectorizedCallSite{T, J, K}, sel::L, args::Vector, ind::Int) where {T, J, K, L <: ConstrainedSelection}
end

function retrace_retained!(vcs::VectorizedCallSite{T, J, K}, sel::L, args::Vector, ind::Int) where {T, J, K, L <: UnconstrainedSelection}
end

function generate_new!(vcs::VectorizedCallSite{T, J, K}, args::Vector, ind::Int) where {T, J, K}
end
