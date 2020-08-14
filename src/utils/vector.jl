# ------------ Vectorized utilities ------------ #

function keyset(sel::L, n_len::Int) where L
    keyset = Set{Int}()
    min = n_len
    for (k, v) in sel.tree
        k isa Int && k > 0 && k <= n_len && !isempty(v) && begin
            push!(keyset, k)
            if k < min
                min = k
            end
        end
    end
    return min, keyset
end

function keyset(sel::Empty, n_len::Int)
    keyset = Set{Int}()
    return n_len, keyset
end

function keyset(sel::L, n_len::Int) where L <: Target
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
