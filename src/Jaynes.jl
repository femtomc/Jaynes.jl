module Jaynes

using IRTools
using IRTools: @dynamo, IR, xcall, arguments, insertafter!, recurse!, isexpr, self, argument!, Variable
using MacroTools
using Distributions

# Differentiable.
using Zygote
using DistributionsAD
using Flux.Optimise: update!
using Flux.Optimise: Descent, ADAM, Momentum, Nesterov, RMSProp, ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW, InvDecay, ExpDecay, WeightDecay
export Descent, ADAM, Momentum, Nesterov, RMSProp, ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW, InvDecay, ExpDecay, WeightDecay

# Toplevel importants :)
const Address = Union{Symbol, Pair{Symbol, Int64}}
import Base.ifelse

# ------------ Com-pirate fixes ------------ #

# TODO: This chunk below me is currently required to fix an unknown performance issue in Base. Don't be alarmed if this suddenly disappears in future versions.
unwrap(gr::GlobalRef) = gr.name
unwrap(gr) = gr

# Whitelist includes vectorized calls.
whitelist = [:rand, 
             :learnable, 
             :markov, 
             :plate, 
             :ifelse, 
             # Foreign model interfaces
             :soss_fmi, :gen_fmi, :turing_fmi]

# Fix for specialized tracing.
function recur!(ir, to = self)
    for (x, st) in ir
        isexpr(st.expr, :call) && begin
            ref = unwrap(st.expr.args[1])
            ref in whitelist || 
            !(unwrap(st.expr.args[1]) in names(Base)) ||
            continue
            ir[x] = Expr(:call, to, st.expr.args...)
        end
    end
    return ir
end

# Fix for _apply_iterate.
function f_push!(arr::Array, t::Tuple{}) end
f_push!(arr::Array, t::Array) = append!(arr, t)
f_push!(arr::Array, t::Tuple) = append!(arr, t)
f_push!(arr, t) = push!(arr, t)
function flatten(t::Tuple)
    arr = Any[]
    for sub in t
        f_push!(arr, sub)
    end
    return arr
end

# ------------ includes ------------ #

using Mjolnir
using Mjolnir: Basic, AType, Const, abstract, Multi, @abstract, Partial, trace
using Mjolnir: Defaults
include("compiler/static.jl")

include("core.jl")
include("selections.jl")
include("learnable.jl")
include("utils/numerical.jl")
include("utils/vectorized.jl")
include("utils/visualization.jl")
include("compiler/diffs.jl")
include("contexts.jl")
include("inference.jl")

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
export Simulate, simulate
export Update, update
export Propose, propose
export Regenerate, regenerate
export Score, score
export Backpropagate, get_parameter_gradients, get_choice_gradients, train

# Tracer language features.
export learnable, plate, markov, ifelse

# Diffs
export NoChange, UndefinedChange, VectorDiff

# Selections.
export selection, get_selection, get_parameters, compare, has_query, update_parameters

# Inference.
export importance_sampling, initialize_filter, filter_step!, metropolis_hastings, resample!, advi

# Utilities.
export display, merge, get_score

end # module
