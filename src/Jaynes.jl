module Jaynes

# Yarrrr I'm a com-pirate!
using IRTools
using IRTools: @dynamo, IR, xcall, arguments, insertafter!, recurse!, isexpr, self, argument!, Variable, meta
#using Mjolnir
#using Mjolnir: Basic, AType, Const, abstract, Multi, @abstract, Partial, trace
#using Mjolnir: Defaults
using MacroTools

using Reexport
@reexport using Distributions
import Distributions: logpdf

# Chainz.
using ZigZagBoomerang
import ZigZagBoomerang: Boomerang, sparse
export Boomerang, sparse
using SparseArrays
using LinearAlgebra
import LinearAlgebra: I
export I

# Differentiable.
@reexport using Zygote
using Zygote.Forward: @tangent
import Zygote.literal_getproperty

# Distances.
@reexport using Distances

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
using Flux.Optimise: update!
@reexport using Flux.Optimise

# Toplevel importants :)
const Address = Union{Int, Symbol, Pair}

import Base.isless
isless(::Symbol, ::Pair) = true
isless(::Pair, ::Symbol) = false
isless(::Int, ::Symbol) = true
isless(::Symbol, ::Int) = false
isless(::Int, ::Pair) = true
isless(::Pair, ::Int) = false

# ------------ Com-pirate fixes ------------ #

# TODO: This chunk below me is currently required to fix an unknown performance issue in Base. Don't be alarmed if this suddenly disappears in future versions.
unwrap(gr::GlobalRef) = gr.name
unwrap(gr) = gr

# Whitelist includes vectorized calls.
whitelist = [
             # Base.
             :rand, :_apply_iterate, :collect,

             # Specialized call sites.
             :markov, :plate, :cond, 

             # Interactions with the context.
             :learnable, :fillable, :factor,

             # Foreign model interfaces.
             :foreign, :deep
            ]

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

include("core.jl")
include("compiler.jl")
include("contexts.jl")
include("inference.jl")
include("language_extensions.jl")
include("utils.jl")

# Contexts.
export Generate, generate
export Simulate, simulate
export Update, update
export Propose, propose
export Regenerate, regenerate
export Score, score
export Backpropagate, get_learnable_gradients, get_choice_gradients, train
export ForwardMode, get_target_gradient

# Tracer language features.
export plate, markov, cond
export learnable, fillable, factor

# Compiler.
export NoChange, UndefinedChange, VectorDiff
export construct_graph, compile_function

# Vectors to dynamic value address map.
export dynamic

# Selections and parameters.
export select, target, array, learnables
export anywhere, intersection, union
export compare, update_learnables, merge!, merge

# Inference.
export maximum_likelihood_estimation, mle
export maximum_a_posteriori_estimation, maxap
export importance_sampling, is
export metropolis_hastings, mh
export hamiltonian_monte_carlo, hmc
export elliptical_slice, es
export piecewise_deterministic_markov_kernel, pdmk
export exchange, ex
export initialize_filter, filter_step!, check_ess_resample!, resample!, get_lmle, pf
export automatic_differentiation_variational_inference, advi
export automatic_differentiation_geometric_vimco, adgv

# Foreign model interfaces.
export @primitive
export @load_gen_fmi
export @load_soss_fmi
export @load_flux_fmi
export @load_advanced_hmc
export @load_chains
export foreign, deep

# Utilities.
export display, getindex, haskey, get_score, get_ret, flatten

# Just a little sugar.
export @sugar

end # module
