# ------------ Executions contexts ------------ #

abstract type ExecutionContext end

# These are "soft" interfaces, not all of these methods apply to every subtype of ExecutionContext.
increment!(ctx::T, w::F) where {T <: ExecutionContext, F <: AbstractFloat} = ctx.weight += w

visit!(ctx::T, addr) where T <: ExecutionContext = visit!(ctx.visited, addr)

get_prev(ctx::T, addr) where T <: ExecutionContext = get_sub(ctx.prev, addr)

function add_choice!(ctx::T, addr, score, v) where T <: ExecutionContext
    ctx.score += score
    set_sub!(ctx.tr, addr, Choice(score, v))
end

function add_value!(ctx::T, addr, score, v) where T <: ExecutionContext
    ctx.score += score
    set_sub!(ctx.map, addr, Value(v))
end

function add_call!(ctx::T, addr, cs::C) where {T <: ExecutionContext, C <: CallSite}
    ctx.score += get_score(cs)
    set_sub!(ctx.tr, addr, cs)
end

# ----------- Control compiler passes with options ------------ #

struct CompilationOptions{AA, Spec} end

const DefaultPipeline= CompilationOptions{:off, :off}
const SpecializePipeline = CompilationOptions{:off, :on}
const AutomaticAddressingPipeline = CompilationOptions{:on, :off}

extract_options(::Type{CompilationOptions{AA, Spec}}) where {AA, Spec} = (AA = AA, Spec = Spec)

# ------------ includes ------------ #

# Generating traces and scoring them according to models.
include("contexts/generate.jl")
include("contexts/simulate.jl")
include("contexts/propose.jl")
include("contexts/assess.jl")

# Used to adjust the score when branches need to be pruned.
function adjust_to_intersection(am::T, visited::V) where {T <: AddressMap, V <: Visitor}
    adj_w = 0.0
    for (k, v) in get_iter(am)
        haskey(visited, k) || begin
            adj_w += get_score(v)
        end
    end
    adj_w
end

include("contexts/update.jl")
include("contexts/regenerate.jl")

# Gradients.
include("contexts/backpropagate.jl")
include("contexts/forwardmode.jl")
