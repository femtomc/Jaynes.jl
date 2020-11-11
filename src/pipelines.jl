# ------------ Executions pipelines ------------ #

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

abstract type CompilationOptions end

struct DefaultCompilationOptions{Spec, AA} <: CompilationOptions end

const DefaultPipeline= DefaultCompilationOptions{:off, :off}
const SpecializerPipeline = DefaultCompilationOptions{:off, :on}
const AutomaticAddressingPipeline = DefaultCompilationOptions{:on, :off}

extract_options(::Type{DefaultCompilationOptions{AA, Spec}}) where {AA, Spec} = (AA = AA, Spec = Spec)

# ------------ includes ------------ #

include("pipelines/gradient_store.jl")
include("pipelines/contexts.jl")
include("pipelines/pipelines.jl")
include("pipelines/gfi/generate.jl")
include("pipelines/gfi/simulate.jl")
include("pipelines/gfi/propose.jl")
include("pipelines/gfi/assess.jl")
include("pipelines/gfi/update.jl")
include("pipelines/gfi/regenerate.jl")
include("pipelines/gfi/backpropagate.jl")
include("pipelines/gfi/forwardmode.jl")

# ------------ Documentation ------------ #
