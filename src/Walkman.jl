module Walkman

using IRTools
using IRTools: IR, @dynamo, recurse!, xcall, self, insertafter!, insert!
using Cassette
using Cassette: recurse
using Distributions
using ExportAll

include("core.jl")
include("trace_context.jl")
include("utils.jl")
include("importance_sampling.jl")

@exportAll()

end # module
