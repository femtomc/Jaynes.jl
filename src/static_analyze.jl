module Jaynes

import Base.rand
import JSON
using MetaGraphs, LightGraphs
using GraphPlot
using Compose
import Cairo

# A bit of IR meta-programming required :)
using IRTools
using IRTools: blocks
using IRTools: @code_ir, @dynamo, IR, recurse!, var, Variable, Statement, isexpr
using Random

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

# Dependency graph.
const DependencyGraph = MetaDiGraph
add_v!(n::Any, g::DependencyGraph) = !(n in [get_prop(g, i, :name) for i in vertices(g)]) && add_vertex!(g, :name, n)
loc_tuple = (depth, x) -> (l = getlocation(x.info); (l.block, l.line))

# Utilities for moving around the IR.
head(p::Pair) = p[1]
tail(p::Pair) = p[2]

function remainder(var::Variable, ir)::Array{Pair{Variable, Statement}}
    map(v -> v => ir[v], filter(x -> x.id > var.id, keys(ir)))
end

function var_check(var::Variable, st::Statement)
    var in st.expr.args && return true
end

function dependents(var::Variable, ir)::Array{Pair{Variable, Statement}}
    rmdr = remainder(var, ir)
    a = Array{Pair{Variable, Statement}}([])
    length(rmdr) == 0 ? nothing : map(x -> var_check(var, tail(x)) ? push!(a, x) : nothing, rmdr)
    return a
end

function dependency_graph(ir)::DependencyGraph
    g = MetaDiGraph()
    set_indexing_prop!(g, :name)
    map(x -> add_v!(x, g), keys(ir))
    for var in keys(ir)
        map(x -> add_edge!(g, g[var, :name], g[head(x), :name]), dependents(var, ir))
    end
    return g
end

function track_rand(ir)::Array{Dict{Any, Any}}
    accounted_for = Array{Variable, 1}([])
    trees = Array{Dict{Any, Any}}([])
    for (var, st) in ir
        if (eval(st.expr.args[1]) isa typeof(rand) && !(var in accounted_for))
            push!(trees, grow_tree(var, ir, Dict{Any, Any}()))
        end
    end
    return trees
end

# A simple example.
function simple(z::Float64)
    x = rand(Normal(z, 1.0))
    y = x + rand(Normal(x, 1.0))
    return y
end

ir = @code_ir simple(5.0)
println(ir)
for (var, st) in ir
    println(dependents(var, ir))
end

mg = dependency_graph(ir)
labels = [(l = get_prop(mg, i, :name); l) for i in vertices(mg)]
draw(PDF("graphs/dependency_graph_irtools.pdf", 16cm, 16cm), gplot(mg, nodelabel = labels, arrowlengthfrac = 0.1, layout=stressmajorize_layout))

end # module
