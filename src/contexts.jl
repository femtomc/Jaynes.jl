abstract type ExecutionContext end

# These are "soft" interfaces, not all of these methods apply to every subtype of ExecutionContext.
increment!(ctx::T, w::Float64) where T <: ExecutionContext = ctx.weight += w
get_sub(ctx::T, addr) where T <: ExecutionContext = get_sub(ctx.select, addr)
get_subparameters(ctx::T, addr) where T <: ExecutionContext = get_sub(ctx.params, addr)
visit!(ctx::T, addr) where T <: ExecutionContext = visit!(ctx.visited, addr)
get_prev(ctx::T, addr) where T <: ExecutionContext = get_call(ctx.prev, addr)
function add_choice!(ctx::T, addr, cs::ChoiceSite) where T <: ExecutionContext
    ctx.score += get_score(cs)
    add_choice!(ctx.tr, addr, cs)
end
function add_call!(ctx::T, addr, cs::CallSite) where T <: ExecutionContext
    ctx.score += get_score(cs)
    add_call!(ctx.tr, addr, cs)
end

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
include("contexts/simulate.jl")
include("contexts/propose.jl")
include("contexts/score.jl")
include("contexts/update.jl")
include("contexts/regenerate.jl")
include("contexts/backpropagate.jl")
