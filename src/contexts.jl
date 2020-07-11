abstract type ExecutionContext end
increment!(ctx::T, w::Float64) where T <: ExecutionContext = ctx.weight += w
get_subselection(ctx::T, addr) where T <: ExecutionContext = get_sub(ctx.select, addr)

@dynamo function (mx::ExecutionContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    recur!(ir)
    return ir
end

function (mx::ExecutionContext)(::typeof(Core._apply_iterate), f, c::typeof(rand), args...)
    return mx(c, flatten(args)...)
end

include("contexts/generate.jl")
include("contexts/propose.jl")
include("contexts/score.jl")
include("contexts/update.jl")
include("contexts/regenerate.jl")
include("contexts/backpropagate.jl")
