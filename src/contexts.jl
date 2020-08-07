abstract type ExecutionContext end

# These are "soft" interfaces, not all of these methods apply to every subtype of ExecutionContext.
increment!(ctx::T, w::Float64) where T <: ExecutionContext = ctx.weight += w
get_subselection(ctx::T, addr) where T <: ExecutionContext = get_sub(ctx.select, addr)
get_subparameters(ctx::T, addr) where T <: ExecutionContext = get_sub(ctx.params, addr)
visit!(ctx::T, addr) where T <: ExecutionContext = visit!(ctx.visited, addr)
get_prev(ctx::T, addr) where T <: ExecutionContext = get_sub(ctx.prev, addr)
function add_choice!(ctx::T, addr, cs::ChoiceSite) where T <: ExecutionContext
    ctx.score += get_score(cs)
    add_choice!(ctx.tr, addr, cs)
end
function add_call!(ctx::T, addr, cs::CallSite) where T <: ExecutionContext
    ctx.score += get_score(cs)
    add_call!(ctx.tr, addr, cs)
end
function add_call!(ctx::T, addr, cs::CallSite, sc::Float64) where T <: ExecutionContext
    ctx.score += get_score(cs) + sc
    add_call!(ctx.tr, addr, cs)
end
function add_call!(ctx::T, cs::CallSite) where T <: ExecutionContext
    ctx.score += get_score(cs)
    # TODO: should only work for VectorizedTraces - make error explicit here.
    add_call!(ctx.tr, cs)
end

@dynamo function (mx::ExecutionContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    recur!(ir)
    return ir
end

(mx::ExecutionContext)(::typeof(Core._apply_iterate), f, c::typeof(rand), args...) = mx(c, flatten(args)...)

# ------------ includes ------------ #

# Generating traces and scoring them according to models.
include("contexts/generate.jl")
include("contexts/simulate.jl")
include("contexts/propose.jl")
include("contexts/score.jl")

# Used to adjust the score when branches need to be pruned.
function adjust_to_intersection(tr::T, visited::V) where {T <: Trace, V <: Visitor}
    adj_w = 0.0
    for (k, v) in dump_top(tr)
        has_top(visited, k) || begin
            adj_w += get_score(v)
        end
    end
    for (k, v) in dump_sub(tr)
        has_sub(visited, k) || begin
            adj_w += get_score(v)
        end
    end
    adj_w
end

include("contexts/update.jl")
include("contexts/regenerate.jl")

# Gradients.
include("contexts/backpropagate.jl")
