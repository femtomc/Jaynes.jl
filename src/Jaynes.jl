module Jaynes

using Cassette
using Cassette: recurse, similarcontext, disablehooks, Reflection
using Cthulhu
using Revise
using MacroTools
using MacroTools: postwalk
using IRTools
using IRTools: meta
using Flux
using Flux: Params
using Zygote
using Distributions
using DistributionsAD
using FunctionalCollections: PersistentVector
using ExportAll

const Address = Union{Symbol, Pair}

include("core/trace.jl")
include("core/context.jl")
include("core/gradients.jl")
include("core/analysis.jl")
include("utils.jl")
include("inference/importance_sampling.jl")
include("inference/particle_filter.jl")
include("inference/inference_compilation.jl")
include("tracing.jl")
include("core/ignore_pass.jl")

function derive_debug(mod; type_tracing = false)
    @assert mod isa Module
    fns = filter(names(mod)) do nm
        try
            Base.eval(mod, nm) isa Function
        catch e
            println("Ignoring call in $e.")
            false
        end
    end
    @eval begin
        import Cassette.prehook
        import Cassette.posthook
        using Revise
    end

    exprs = map(fns) do f
        if type_tracing
            @eval begin
                function prehook(::Jaynes.TraceCtx, call::typeof($mod.$f), args...)
                    @info "$(stacktrace()[3])\n" call typeof(args)
                    println("Beginning type inference...")
                    Cthulhu.descend(call, typeof(args))
                end
            end
        else
            @eval begin
                function prehook(::Jaynes.TraceCtx, call::typeof($mod.$f), args...)
                    @info "$(stacktrace()[3])\n" call typeof(args)
                end
            end
        end
    end
end

@exportAll()

end # module
