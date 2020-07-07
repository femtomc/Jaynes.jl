# Vectorized utilities.
function keyset(sel::L, new_length::Int) where L <: ConstrainedSelection
    keyset = Set{Int}()
    for (k, _) in sel.tree
        k isa Int && k > 0 && k <= n_len && push!(keyset, k)
    end
    return keyset
end

function keyset(sel::L, new_length::Int) where L <: UnconstrainedSelection
    keyset = Set{Int}()
    for (k, _) in sel.tree
        k isa Int && k > 0 && k <= n_len && push!(keyset, k)
    end
    return keyset
end

function score_adj(vcs::VectorizedCallSite{T, J, K}, o_len::Int, n_len::Int) where {T, J, K, L <: Selection}
    score_acc = 0.0
    for k in n_len + 1 : o_len
        sub = vcs.subtraces[k]
        score_acc += score(sub)
    end
    return score_acc
end

