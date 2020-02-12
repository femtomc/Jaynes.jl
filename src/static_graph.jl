module StaticGraph

import Base.rand
using IRTracker
using LightGraphs, MetaGraphs

# Source of randomness with the right methods.
abstract type Randomness end
struct Normal <: Randomness
    name::Symbol
    μ::Float64
    σ::Float64
    Normal(μ::Float64, σ::Float64) = new(gensym(), μ, σ)
end
rand(x::Normal) = x.μ + rand()*x.σ
logprob(x::Normal, pt::Float64) = -(1.0/2.0)*( (pt-x.μ)/x.σ )^2 - (log(x.σ) + log(2*pi))

# Recurse through the nested IR graph produced by IRTracker.
function recurse_children(x::T, store::Dict, depth::Int, depth_lim::Int) where T <: AbstractNode
    if depth > depth_lim
        return store
    else
        children = getchildren(x)
        calls = filter(i -> i isa NestedCallNode, children)
        rands = filter(k -> k.call.f.value isa typeof(rand), calls)
        map(x -> println(dependents(x)), rands)
        map(x -> recurse_children(x, store, depth + 1, depth_lim), rands)
    end
end

recurse_children(x::T, depth_lim::Int) where T <: AbstractNode = recurse_children(x, Dict(), 0, depth_lim)

# A simple example.
function simple(z::Float64)
    x = rand(Normal(z, 1.0))
    y = x + rand(Normal(x, 1.0))
    return y
end

en_ir = track(simple, 5.0)
store = recurse_children(en_ir, 3)
end
