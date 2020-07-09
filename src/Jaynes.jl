module Jaynes

using IRTools
using IRTools: @dynamo, IR, xcall, arguments, insertafter!, recurse!, isexpr, self, argument!, Variable
using MacroTools
using Distributions
using DistributionsAD
using Zygote
using Flux.Optimise
using Mjolnir
using Mjolnir: Basic, AType, Const, abstract, Multi, @abstract, Partial
using Mjolnir: Defaults
import Mjolnir: trace

# Toplevel importants :)
abstract type ExecutionContext end
const Address = Union{Symbol, Pair{Symbol, Int64}}

include("compiler/static.jl")
include("trace.jl")
include("selections.jl")
include("utils/numerical.jl")
include("utils/vectorized.jl")
include("utils/visualization.jl")
include("compiler/diffs.jl")
include("contexts/contexts.jl")

# Utility structure for collections of samples.
mutable struct Particles{C}
    calls::Vector{C}
    lws::Vector{Float64}
    lmle::Float64
end

include("inference/importance_sampling.jl")
include("inference/particle_filtering.jl")
include("inference/metropolis_hastings.jl")

# Foreign models.
include("foreign_model_interfaces/blackbox.jl")
export @primitive

include("foreign_model_interfaces/soss.jl")
export @load_soss_fmi

include("foreign_model_interfaces/gen.jl")
export @load_gen_fmi

include("foreign_model_interfaces/turing.jl")
export @load_turing_fmi

# Contexts.
export Generate, generate
export Update, update
export Propose, propose
export Regenerate, regenerate
export Score, score
export Backpropagate, get_parameter_gradients, get_choice_gradients

# Trace.
export Trace, trace, get_score, learnable

# Selections.
export selection, get_selection, get_parameters, compare, has_query, update!

# Inference.
export importance_sampling, initialize_filter, filter_step!, metropolis_hastings, resample!

# Utilities.
export display, merge

end # module
