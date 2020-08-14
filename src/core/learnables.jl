# ------------ Gradients and parameters ------------ #

const Gradients = DynamicMap{Value}

@inline function accumulate!(gs::Gradients, addr, v)
    set_sub!(gs, addr, Value(v + get(gs, addr, 0.0)))
end

@inline function accumulate!(gs::Gradients, addr, v::Value)
    set_sub!(gs, addr, Value(get_value(v) + get(gs, addr, 0.0)))
end

@inline function accumulate!(gs1::Gradients, gs2::Gradients)
    for k in setdiff(keys(gs2.tree), keys(gs1.tree))
        accumulate!(gs1, k, get_sub(gs2, k))
    end
    inter = intersect(keys(gs1.tree), keys(gs2.tree))
    for k in inter
        accumulate!(gs1, k, get_sub(gs2, k))
    end
end

const LearnableByAddress = DynamicMap{Value}
Parameters() = LearnableByAddress()

function learnables(arr::Array{Pair{T, K}}) where {T <: Tuple, K}
    top = LearnableByAddress()
    map(arr) do (k, v)
        set_sub!(top, k, Value(v))
    end
    return top
end

function learnables(p::Pair{T, K}) where {T <: Symbol, K <: AddressMap}
    top = LearnableByAddress()
    set_sub!(top, p[1], p[2])
    return top
end

# ------------ update_learnables links into Flux optimiser APIs ------------ #

function update_learnables(opt, a::LearnableByAddress, b::Gradients)
    p_arr = array(a, Float64)
    gs_arr = array(b, Float64)
    update!(opt, p_arr, -gs_arr)
    return target(a, p_arr)
end
