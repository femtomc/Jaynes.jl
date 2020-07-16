# Vectorized utilities.
function keyset(sel::L, n_len::Int) where L <: ConstrainedSelection
    keyset = Set{Int}()
    min = n_len
    for (k, _) in sel.tree
        k isa Int && k > 0 && k <= n_len && begin
            push!(keyset, k)
            if k < min
                min = k
            end
        end
    end
    return min, keyset
end

function keyset(sel::L, n_len::Int) where L <: UnconstrainedSelection
    keyset = Set{Int}()
    min = n_len
    for (k, _) in sel.tree
        k isa Int && k > 0 && k <= n_len && begin
            push!(keyset, k)
            if k < min
                min = k
            end
        end
    end
    return min, keyset
end
