module Jaynes

using Cthulhu

# IRRRR I'm a com-pirate.
using Cassette
using Cassette: recurse, similarcontext, disablehooks, Reflection, canrecurse
import Cassette: overdub, prehook, posthook, Reflection
using MacroTools
using MacroTools: postwalk
using IRTools
using IRTools: meta, IR, slots!
import IRTools: meta, IR
using Mjolnir

using Distributions

using Flux
using Flux: Params
using Zygote
using DistributionsAD
using ExportAll

const Address = Union{Symbol, Pair}

include("core/selections.jl")
include("core/trace.jl")
include("core/contexts.jl")
include("core/gradients.jl")
include("core/blackbox.jl")
include("core/language_cores.jl")
include("utils.jl")
include("inference/importance_sampling.jl")
include("inference/particle_filter.jl")
include("inference/inference_compilation.jl")
include("tracing.jl")
include("core/passes.jl")

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
        using Revise
    end

    exprs = map(fns) do f
        if type_tracing
            @eval begin
                function Jaynes.prehook(::Jaynes.TraceCtx, call::typeof($mod.$f), args...)
                    @info "$(stacktrace()[3])\n" call typeof(args)
                    println("Beginning type inference...")
                    Cthulhu.descend(call, typeof(args))
                end
            end
        else
            @eval begin
                function Jaynes.prehook(::Jaynes.TraceCtx, call::typeof($mod.$f), args...)
                    @info "$(stacktrace()[3])\n" call typeof(args)
                end
            end
        end
    end
end


@exportAll()

end # module
