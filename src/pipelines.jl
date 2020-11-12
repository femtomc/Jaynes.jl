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

struct DefaultCompilationOptions{H, S, Sp, AA} <: CompilationOptions end

const DefaultPipeline= DefaultCompilationOptions{true, true, false, false}
const SpecializerPipeline = DefaultCompilationOptions{true, true, true, false}
const AutomaticAddressingPipeline = DefaultCompilationOptions{true, true, false, true}

extract_options(::Type{DefaultCompilationOptions{H, S, Sp, AA}}) where {H, S, Sp, AA} = (H = H, S = S, Sp = Sp, AA = AA)

# ------------ includes ------------ #

include("pipelines/gradient_store.jl")
include("pipelines/contexts.jl")
include("pipelines/default_pipeline.jl")
include("pipelines/gfi/generate.jl")
include("pipelines/gfi/simulate.jl")
include("pipelines/gfi/propose.jl")
include("pipelines/gfi/assess.jl")
include("pipelines/gfi/update.jl")
include("pipelines/gfi/regenerate.jl")
include("pipelines/gfi/backpropagate.jl")
include("pipelines/gfi/forwardmode.jl")

# ------------ Documentation ------------ #
