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
include("update.jl")
include("regenerate.jl")
