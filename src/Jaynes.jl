module Jaynes

# Yarrrr I'm a com-pirate!
using IRTools
using IRTools: @dynamo, IR, xcall, arguments, insertafter!, recurse!, isexpr, self, argument!, Variable
using Mjolnir
using Mjolnir: Basic, AType, Const, abstract, Multi, @abstract, Partial, trace
using Mjolnir: Defaults
using MacroTools

using Reexport
@reexport using Distributions

# Differentiable.
using Zygote
using DistributionsAD
using Flux.Optimise: update!
@reexport using Flux.Optimise

# Toplevel importants :)
const Address = Union{Int, Symbol, Pair}

# ------------ Com-pirate fixes ------------ #

# TODO: This chunk below me is currently required to fix an unknown performance issue in Base. Don't be alarmed if this suddenly disappears in future versions.
unwrap(gr::GlobalRef) = gr.name
unwrap(gr) = gr

# Whitelist includes vectorized calls.
whitelist = [:rand, 
             :learnable, 
             :markov, 
             :plate, 
             :cond, 
             :_apply_iterate,
             # Foreign model interfaces
             :soss_fmi, :gen_fmi, :turing_fmi]

# Fix for specialized tracing.
function recur!(ir, to = self)
    for (x, st) in ir
        isexpr(st.expr, :call) && begin
            ref = unwrap(st.expr.args[1])
            ref in whitelist || continue
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

include("compiler/static.jl")
include("traces.jl")
include("selections.jl")
include("utils/numerical.jl")
include("utils/vectorized.jl")
include("compiler/diffs.jl")
include("contexts.jl")
include("inference.jl")
include("foreign_model_interfaces.jl")

# Contexts.
export Generate, generate
export Simulate, simulate
export Update, update
export Propose, propose
export Regenerate, regenerate
export Score, score
export Backpropagate, get_parameter_gradients, get_choice_gradients, train

# Tracer language features.
export learnable, plate, markov, cond

# Diffs.
export NoChange, UndefinedChange, VectorDiff

# Selections and parameters.
export selection, array, parameters
export anywhere, intersection, union
export get_selection, compare, has_top, update_parameters, dump_queries, merge!, merge

# Inference.
export importance_sampling, is
export initialize_filter, filter_step!, check_ess_resample!, get_lmle, pf
export metropolis_hastings, mh
export automatic_differentiation_variational_inference, advi
export hamiltonian_monte_carlo, hmc

# Foreign model interfaces.
export @primitive
export @load_gen_fmi, gen_fmi
export @load_soss_fmi, soss_fmi

# Utilities.
export display, getindex, haskey, get_score, get_ret

end # module
