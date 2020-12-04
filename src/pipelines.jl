# ------------ Execution pipelines ------------ #

# This abstract type represents a "compiler execution context". 
# This context is allowed to look at certain calls, and re-interpret them using dispatch.
# This allows the expression of tracing as a tracked side-effect.

abstract type ExecutionContext{J, K, T} end

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

const NoStatic = DefaultCompilationOptions{false, false, false, false}
const DefaultPipeline = DefaultCompilationOptions{false, true, false, false}
const StaticWithLints = DefaultCompilationOptions{true, true, false, false}
const SpecializerPipeline = DefaultCompilationOptions{true, true, true, false}
const AutomaticAddressingPipeline = DefaultCompilationOptions{true, true, false, true}

extract_options(::Type{DefaultCompilationOptions{H, S, Sp, AA}}) where {H, S, Sp, AA} = (H = H, S = S, Sp = Sp, AA = AA)

# ------------ includes ------------ #

# Note - includes must come before the dynamo definitions because of world age issues.

include("pipelines/gradient_store.jl")
include("pipelines/contexts.jl")

const DiffAware = Union{UpdateContext, RegenerateContext}
const DoesNotCare = Union{SimulateContext, GenerateContext, AssessContext, ProposeContext, ForwardModeContext, BackpropagationContext}

include("pipelines/default_pipeline.jl")
include("pipelines/gfi/generate.jl")
include("pipelines/gfi/simulate.jl")
include("pipelines/gfi/propose.jl")
include("pipelines/gfi/assess.jl")
include("pipelines/gfi/update.jl")
include("pipelines/gfi/regenerate.jl")
include("pipelines/gfi/backpropagate.jl")
include("pipelines/gfi/forwardmode.jl")

# Staging for "doesn't care about incremental diff" contexts.
# These basically deal with straightforward execution - they don't require us to think about incremental computing.
@dynamo function (sx::ExecutionContext{J, K, T})(a...) where {J, K, T <: DoesNotCare}
    ir = IR(a...)
    ir == nothing && return
    ir = staged_pipeline(ir, J, K, T)
    ir
end

# Diff aware contexts are a bit tricky.
# Sneaky invoke hackz. Allows you to pass in arguments with types which don't match the dispatch table. Dangerous - but required for incremental computation on incremental computation unaware code.
@dynamo function (sx::ExecutionContext{J, K, T})(f, ::Type{S}, args...) where {J, S <: Tuple, K, T <: DiffAware}
    ir = IR(f, S.parameters...)
    ir == nothing && return
    ir = staged_pipeline(ir, J, K, UpdateContext)
    ir
end

# Fixes for Base.
function (sx::ExecutionContext)(::typeof(Core._apply_iterate), f, c::typeof(trace), args...)
    flt = flatten(args)
    addr, rest = flt[1], flt[2 : end]
    ret, cl = simulate(rest...)
    add_call!(sx, addr, cl)
    ret
end

function (sx::ExecutionContext)(::typeof(Base.collect), generator::Base.Generator)
    map(generator.iter) do i
        sx(generator.f, i)
    end
end

function (sx::ExecutionContext)(::typeof(Base.map), fn, iter)
    map(iter) do i
        sx(fn, i)
    end
end

function (sx::ExecutionContext)(::typeof(Base.filter), fn, iter)
    filter(iter) do i
        sx(fn, i)
    end
end

# ------------ Documentation ------------ #
