# Address map interfaces
function fill_array!(val::T, arr::Vector{T}, f_ind::Int) where T
    if length(arr) < f_ind
        resize!(arr, 2 * f_ind)
    end
    arr[f_ind] = val
    1
end

function fill_array!(val::Vector{T}, arr::Vector{T}, f_ind::Int) where T
    if length(arr) < f_ind + length(val)
        resize!(arr, 2 * (f_ind + length(val)))
    end
    arr[f_ind : f_ind + length(val) - 1] = val
    length(val)
end

function fill_array!(val::Value, arr::Vector{T}, f_ind::Int) where T
    fill_array!(get_value(val), arr, f_ind)
end

function fill_array!(val::Choice, arr::Vector{T}, f_ind::Int) where T
    fill_array!(get_value(val), arr, f_ind)
end

function fill_array!(val::T, arr::Vector{K}, f_ind::Int) where {K, T <: AddressMap}
    sorted_keys = sort(collect(keys(shallow_iterator(val))))
    idx = f_ind
    for k in sorted_keys
        n = fill_array!(get_sub(val, k), arr, idx)
        idx += n
    end
    idx - f_ind
end

function fill_array!(val::Vector{T}, arr::Vector{T}, f_ind::Int) where T
    if length(arr) < f_ind + length(val)
        resize!(arr, 2 * (f_ind + length(val)))
    end
    arr[f_ind : f_ind + length(val) - 1] = val
    length(val)
end

function array(gs::K, ::Type{T}) where {T, K}
    arr = Vector{T}(undef, 32)
    n = fill_array!(gs, arr, 1)
    resize!(arr, n)
    arr
end

function from_array(::T, arr::Vector{T}, f_ind::Int) where T
    (1, arr[f_ind])
end

function from_array(val::Vector{T}, arr::Vector{T}, f_ind::Int) where {T}
    n_read = length(val)
    (n_read, arr[f_ind : f_ind + n_read - 1])
end

function from_array(::SelectAll, arr::Vector{K}, f_ind::Int) where K
    (1, Value(arr[f_ind]))
end

function from_array(v::Value, arr::Vector{K}, f_ind::Int) where K
    shape = get_value(v)
    n, v = from_array(shape, arr, f_ind)
    (n, Value(v))
end

function from_array(c::Choice, arr::Vector{K}, f_ind::Int) where K
    shape = get_value(c)
    n, v = from_array(shape, arr, f_ind)
    (n, Value(v))
end

function from_array(schema::T, arr::Vector{K}, f_ind::Int) where {K, T <: DynamicMap}
    sel = DynamicMap{Value}()
    sorted_keys = sort(collect(keys(shallow_iterator(schema))))
    idx = f_ind
    for k in sorted_keys
        (n, v) = from_array(get_sub(schema, k), arr, idx)
        idx += n
        set_sub!(sel, k, v)
    end
    (idx - f_ind, sel)
end

function target(schema::K, arr::Vector) where K
    (n, sel) = from_array(schema, arr, 1)
    n != length(arr) && error("Dimension error: length of arr $(length(arr)) must match $n.")
    sel
end
