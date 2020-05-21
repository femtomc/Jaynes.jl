module Jaynes

using IRTools
using IRTools: IR, @dynamo, recurse!, xcall, self, insertafter!, insert!
using Distributions
using ExportAll

include("core.jl")
include("trace_dynamo.jl")
include("utils.jl")
include("importance_sampling.jl")

@exportAll()

end # module
