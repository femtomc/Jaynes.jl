module DynamicGraph

import Base.rand
using IRTracker
using LightGraphs, MetaGraphs
using GraphPlot
using Compose
import Cairo

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

const DependencyGraph = MetaDiGraph
add_v!(n::Any, g::DependencyGraph) = !(n in [get_prop(g, i, :name) for i in vertices(g)]) && add_vertex!(g, :name, n)
loc_tuple = (depth, x) -> (l = getlocation(x.info); (l.block, l.line))

# Recurse through the nested IR graph produced by IRTracker.
function recurse_children(x::T, g::DependencyGraph, depth::Int, depth_lim::Int) where T <: AbstractNode
    if depth > depth_lim
        return

    else
        children = getchildren(x)
        calls = filter(i -> i isa NestedCallNode, children)
        rands = filter(k -> getvalue(k.call.f) isa typeof(rand), calls)
        map(x -> (
                  add_v!(loc_tuple(depth, x), g);
                  depends = forward(x);
                  map(d -> add_v!(loc_tuple(depth, d), g), depends);
                  map(d -> add_edge!(g, g[loc_tuple(depth, x), :name], g[loc_tuple(depth, d), :name]), depends)
                 ), rands)
        return g
    end
end

recurse_children(x::T, depth_lim::Int) where T <: AbstractNode = (G = MetaDiGraph(); set_indexing_prop!(G, :name); recurse_children(x, G, 0, depth_lim))

# A simple example.
function simple(z::Float64)
    x = rand(Normal(z, 1.0))
    y = x + rand(Normal(x, 1.0))
    if y > 1
        z = rand(Normal(x, 1.0))
    end

    while rand(Normal(z, 10.0)) < 5
        y += x
    end

    return y
end

en_ir = track(simple, 5.0)
printlevels(en_ir, 2)
dg = recurse_children(en_ir, 3)
labels = [(l = get_prop(dg, i, :name); l) for i in vertices(dg)]
draw(PDF("graphs/dependency_graph_irtracker.pdf", 16cm, 16cm), gplot(dg, nodelabel = labels, arrowlengthfrac = 0.1, layout=circular_layout))
end
