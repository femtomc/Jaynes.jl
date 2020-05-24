module Walkman

using Cassette
using Cassette: recurse, similarcontext, disablehooks
using Distributions
using ExportAll

const Address = Union{Symbol, Pair}

include("trace_core.jl")
include("trace_context.jl")
include("utils.jl")
include("importance_sampling.jl")

@exportAll()

end # module
