module Jaynes

using IRTools
using IRTools: @dynamo, IR, xcall, arguments, insertafter!, recurse!, isexpr, self, argument!, Variable
using MacroTools
using Distributions

# Toplevel importants :)
abstract type ExecutionContext end
const Address = Union{Symbol, Pair{Symbol, Int64}}

include("static.jl")
include("trace.jl")
include("selections.jl")
include("utils/numerical.jl")
include("utils/vectorized.jl")
include("utils/visited.jl")
include("utils/visualization.jl")
include("contexts/contexts.jl")

# Utility structure for collections of samples.
mutable struct Particles
    calls::Vector{CallSite}
    lws::Vector{Float64}
    lmle::Float64
end

include("inference/importance_sampling.jl")
include("inference/particle_filtering.jl")
include("inference/metropolis_hastings.jl")
include("blackbox.jl")

# Convenience function for unconstrained generation.
function trace(fn::Function, args...)
    tr = Trace()
    ret = tr(fn, args...)
    return CallSite(tr, fn, args, ret)
end

# Contexts.
export Generate, Update, Propose, Regenerate, Score

# Trace.
export Trace, trace

# Selections.
export selection, compare

# Inference.
export importance_sampling, initialize_filter, filter_step!, metropolis_hastings, resample!

# Utilities.
export display, merge

# Black-box extension
export @primitive

end # module
