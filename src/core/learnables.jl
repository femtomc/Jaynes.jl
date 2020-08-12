# ------------ Gradients and parameters ------------ #

const Gradients = DynamicMap{Value}
const LearnableByAddress = DynamicMap{Value}
Parameters() = LearnableByAddress()

function learnables(arr::Array{Pair{T, K}}) where {T <: Tuple, K}
    top = LearnableByAddress()
    map(arr) do (k, v)
        push!(top, k, v)
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
    return selection(a, p_arr)
end

# ------------ Learnable anywhere ------------ #

const LearnableAnywhere = AnywhereMap{Value}
