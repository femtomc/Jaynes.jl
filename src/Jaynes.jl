module Jaynes

using Cassette
using Cassette: recurse, similarcontext, disablehooks, Reflection
using MacroTools
using MacroTools: postwalk
using IRTools
using Distributions
using FunctionalCollections: PersistentVector
using ExportAll

const Address = Union{Symbol, Pair}

include("trace_core.jl")
include("trace_context.jl")
include("utils.jl")
include("importance_sampling.jl")
include("particle_filter.jl")
include("inference_interfaces.jl")
include("effects.jl")
include("ignore_pass.jl")

@exportAll()

end # module
