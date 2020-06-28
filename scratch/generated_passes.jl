module GeneratedPasses

include("../src/Jaynes.jl")
using .Jaynes
using Cassette
using IRTools
using MacroTools
using MacroTools: unblock, rmlines
using Distributions

Cassette.@context DefaultCtx

# Noop pass.
some_transform(ir) = ir

abstract type Argdiff end
struct NoChange <: Argdiff end
struct Change <: Argdiff end

# Not used right now.
struct Frame
    argdiffs::Tuple{Argdiff}
    dependents::Vector{Symbol}
end

struct UpdateMeta
    compiled::Dict{Tuple{Vararg{Argdiff}}, Function}
    cached::Dict{Function, Tuple}
    UpdateMeta() = new(Dict{Tuple{Vararg{Argdiff}}, Function}(), Dict{Function, Tuple}())
end

function compute_argdiffs(old_args::Tuple{Vararg{K, N}}, new_args::Tuple{Vararg{K, N}}) where {K, N}
    diffs = map(zip(old_args, new_args)) do (o, n)
        if o == n
            NoChange()
        else
            Change()
        end
    end
    Tuple(diffs)
end

# Simple overdub which implements an IR pass in the middle.
function Cassette.overdub(ctx::DefaultCtx, fn::Function, args...)

    # Arg types for introspection.
    t_args = map(args) do a
        typeof(a)
    end

    # Get argdiffs
    if haskey(ctx.metadata.cached, fn)
        argdiffs = compute_argdiffs(ctx.metadata.cached[fn], args)
        println(argdiffs)
        
        # If in cache, grab.
        if haskey(ctx.metadata.compiled, argdiffs)
            fn = ctx.metadata.compiled[argdiffs]
            ret = fn(nothing, args...)

        # Else, transform.
        else
            ir = Jaynes.lower_to_ir(fn, t_args...)
            ir = some_transform(ir)
            println(ir)
            fn = IRTools.func(Main, ir)

            # Cache.
            ctx.metadata.compiled[argdiffs] = fn
            ret = Base.invokelatest(fn, nothing, args...)
        end
    else
        ctx.metadata.compiled[Tuple([NoChange() for _ in args])] = fn
        ctx.metadata.cached[fn] = args
        ret = fn(args...)
    end

    return ret
end

function foo(z::Float64, q::Float64)
    x = rand(:x, Normal(0.0, z))
    y = rand(:y, Normal(x, q))
    return y
end

ctx = DefaultCtx(metadata = UpdateMeta())
ret = Cassette.overdub(ctx, foo, 5.0, 10.0)
ret = Cassette.overdub(ctx, foo, 5.0, 7.0)
println(ret)
println(ctx.metadata)

end # module
