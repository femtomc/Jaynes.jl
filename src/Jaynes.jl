module Jaynes

using Cassette
using Cassette: recurse, similarcontext, disablehooks, Reflection
using MacroTools
using MacroTools: postwalk
using Flux
using Flux: Params
using Distributions
using DistributionsAD
using FunctionalCollections: PersistentVector
using ExportAll

const Address = Union{Symbol, Pair}

include("core/trace.jl")
include("core/context.jl")
include("utils.jl")
include("inference/importance_sampling.jl")
include("inference/particle_filter.jl")
include("inference/inference_compilation.jl")
include("tracing.jl")
include("core/effects.jl")
include("core/ignore_pass.jl")

@exportAll()

end # module
