module Jaynes

using Reexport

import Base.display
import Base: getindex, haskey, iterate, isempty, convert, collect, getindex, setindex!, push!, merge, merge!, get, filter, length, ndims, keys, +, rand, size
import Base: isless
import Base: Pair

# Jaynes implements the abstract GFI from Gen.
import Gen
import Gen: Selection, ChoiceMap, Trace, GenerativeFunction
import Gen: DynamicChoiceMap, EmptySelection
import Gen: get_value, has_value
import Gen: get_values_shallow, get_submaps_shallow
import Gen: get_args, get_retval, get_choices, get_score, get_gen_fn, has_argument_grads, accepts_output_grad, get_params
import Gen: select, choicemap
import Gen: simulate, generate, project, propose, assess, update, regenerate
import Gen: init_param!, accumulate_param_gradients!, choice_gradients, init_update_state, apply_update!

# Gen diff types.
import Gen: Diff, UnknownChange, NoChange
import Gen: SetDiff, DictDiff, VectorDiff, IntDiff, Diffed

# Yarrrr I'm a com-pirate!
using MacroTools
using IRTools
using IRTools: @dynamo, IR, xcall, arguments, insertafter!, recurse!, isexpr, self, argument!, Variable, meta, renumber, Pipe, finish, blocks, predecessors, dominators, block, successors, Block, block!, branches, Branch, branch!, CFG, stmt
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

# ------------ Toplevel importants ------------ #

const Address = Union{Int, Symbol, Pair}

# This is primarily used when mapping choice maps to arrays.
isless(::Symbol, ::Pair) = true
isless(::Pair, ::Symbol) = false
isless(::Int, ::Symbol) = true
isless(::Symbol, ::Int) = false
isless(::Int, ::Pair) = true
isless(::Pair, ::Int) = false

include("unwrap.jl")

# ------------ Com-pirate fixes ------------ #

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

# Jaynes introduces a new type of generative function.
abstract type TypedGenerativeFunction{N, R, Tr, T} <: GenerativeFunction{R, Tr} end

# ------------ includes ------------ #

include("core.jl")
export trace

include("compiler.jl")
export Δ, Diffed, forward
export NoChange, Change
export ScalarDiff, IntDiff, DictDiff, SetDiff, VectorDiff, BoolDiff
export pushforward, _pushforward
export generate_graph_ir
export TraceDefaults
export prepare_ir!, infer!

include("pipelines.jl")
export DefaultPipeline, SpecializerPipeline, AutomaticAddressingPipeline
export record_cached!

include("language_extensions.jl")
export @primitive

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

# Gen compat.
include("gen_fn_interface.jl")

export @jaynes
export JFunction, JTrace
export get_analysis, get_ir
export init_param!, accumulate_param_gradients!, choice_gradients
export choicemap, select
export get_value, has_value
export get_params_grad

# Contexts.
export generate
export simulate
export update
export propose
export regenerate
export assess
export get_learnable_gradients, get_choice_gradients
export get_learnable_gradient, get_choice_gradient

constrain(v::Vector{Pair{T, K}}) where {T <: Tuple, K} = JChoiceMap(target(v))
export constrain

# Typing rules.
include("typing_rules.jl")

# Utilities.
export display, getindex, haskey, get_score, get_ret, flatten, lineplot

end # module
