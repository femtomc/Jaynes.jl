module Jaynes

using IRTools
using IRTools: @dynamo, IR, xcall, arguments, insertafter!, recurse!, isexpr, self, argument!
using Distributions

# Toplevel importants :)
abstract type ExecutionContext end
const Address = Union{Symbol, Pair{Symbol, Int64}}

include("trace.jl")
include("selections.jl")
include("contexts.jl")
include("utils.jl")
include("inference/importance_sampling.jl")
include("inference/particle_filtering.jl")
include("inference/metropolis_hastings.jl")

function call(tr::Trace, fn::Function, args...)
    ret = tr(fn, args...)
    return CallSite(tr, fn, args, ret)
end

# Contexts.
export Generate, Update, Propose, Regenerate, Score

# Trace.
export Trace

# Selections.
export selection, compare

# Inference.
export importance_sampling, initialize_filter, filter_step!, metropolis_hastings

# Utilities.
export display, merge

end # module
