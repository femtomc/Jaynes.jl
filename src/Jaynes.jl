module Jaynes

using Reexport

# Jaynes implements the abstract GFI from Gen.
import Gen
import Gen: Selection, ChoiceMap, Trace, GenerativeFunction
import Gen: DynamicChoiceMap, EmptySelection
import Gen: get_value, has_value
import Gen: get_values_shallow, get_submaps_shallow
import Gen: get_args, get_retval, get_choices, get_score, get_gen_fn, has_argument_grads, accepts_output_grad, get_params
import Gen: select, choicemap
import Gen: simulate, generate, project, propose, assess, update, regenerate
import Gen: init_param!, accumulate_param_gradients!, choice_gradients

# Gen diff types.
import Gen: Diff, UnknownChange, NoChange
import Gen: SetDiff, DictDiff, VectorDiff, IntDiff, Diffed

# Yarrrr I'm a com-pirate!
using MacroTools
using IRTools
using IRTools: @dynamo, IR, xcall, arguments, insertafter!, recurse!, isexpr, self, argument!, Variable, meta, renumber, Pipe, finish, blocks, predecessors, dominators, block, successors, Block, block!, branches, Branch, branch!
using Random
using Mjolnir
using Mjolnir: Basic, AType, Const, abstract, Multi, @abstract, Partial, Node
using Mjolnir: Defaults
using InteractiveUtils: subtypes

# Static selektor.
using StaticArrays

@reexport using Distributions
import Distributions: Distribution
import Distributions: logpdf

# Differentiable.
@reexport using Zygote
using ForwardDiff
using ForwardDiff: Dual
import Zygote.literal_getproperty

# Fix for: https://github.com/FluxML/Zygote.jl/issues/717
Zygote.@adjoint function literal_getproperty(x, ::Val{f}) where f
    val = getproperty(x, f)
    function back(Δ)
        Zygote.accum_param(__context__, val, Δ) # === nothing && return
        if isimmutable(x)
            ((;Zygote.nt_nothing(x)..., Zygote.pair(Val(f), Δ)...), nothing)
        else
            dx = Zygote.grad_mut(__context__, x)
            dx[] = (;dx[]...,Zygote.pair(Val(f), Zygote.accum(getfield(dx[], f), Δ))...)
            return (dx, nothing)
        end
    end
    unwrap(val), back
end

using DistributionsAD

# Plotting.
using UnicodePlots: lineplot

# Toplevel importants :)
const Address = Union{Int, Symbol, Pair}

isless(::Symbol, ::Pair) = true
isless(::Pair, ::Symbol) = false
isless(::Int, ::Symbol) = true
isless(::Symbol, ::Int) = false
isless(::Int, ::Pair) = true
isless(::Pair, ::Int) = false

# ------------ Com-pirate fixes ------------ #

# TODO: This chunk below me is currently required to fix an unknown performance issue in Base. Don't be alarmed if this suddenly disappears in future versions.
unwrap(gr::GlobalRef) = gr.name
unwrap(v::Val{K}) where K = K
unwrap(gr) = gr

# Whitelist includes vectorized calls.
whitelist = [
             # Base.
             :trace, :_apply_iterate, :collect,

             # Interactions with the context.
             :learnable, :fillable, :factor,
            ]

# Fix for specialized tracing.
function recur(ir, to = self)
    pr = Pipe(ir)
    for (x, st) in pr
        isexpr(st.expr, :call) && begin
            ref = unwrap(st.expr.args[1])
            ref in whitelist || continue
            pr[x] = Expr(:call, to, st.expr.args...)
        end
    end
    finish(pr)
end

# Fix for _apply_iterate (used in contexts).
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

# Jaynes.
include("core.jl")
export trace

include("compiler.jl")
export Δ, Diffed, forward
export NoChange, Change
export ScalarDiff, IntDiff, DictDiff, SetDiff, VectorDiff, BoolDiff
export pushforward, _pushforward

include("contexts.jl")
export record_cached!

include("language_extensions.jl")
include("utils.jl")

# Contexts.
export generate
export simulate
export update
export propose
export regenerate
export score
export get_learnable_gradients, get_choice_gradients
export get_learnable_gradient, get_choice_gradient

# Tracer language features.
export learnable, fillable, factor

# Compiler.
export NoChange, UndefinedChange
export construct_graph, compile_function

# Vectors to dynamic value address map.
export dynamic

# Selections and parameters.
export select, target, static, array, learnables
export anywhere, intersection, union
export compare, update_learnables, merge!, merge

# Foreign model interfaces.
export @primitive
export @load_chains
export foreign, deep

# Utilities.
export display, getindex, haskey, get_score, get_ret, flatten, lineplot

# Just a little sugar.
export @sugar

# Gen compat.
import Base.display
include("gen_fn_interface.jl")

export @jaynes
export JFunction, JTrace
export get_analysis
export init_param!, accumulate_param_gradients!, choice_gradients
export choicemap, select
export get_value, has_value

end # module
