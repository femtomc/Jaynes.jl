module Jaynes

import Base.rand
import JSON
import Cairo
using MetaGraphs, LightGraphs
using GraphPlot
using Compose

# A bit of meta-programming required :)
using MacroTools
using IRTools
using IRTools: blocks
using IRTools: @code_ir, @dynamo, IR, recurse!, Variable
using Random

# Source of randomness with the right methods.
abstract type Randomness end
struct Normal <: Randomness
    μ::Float64
    σ::Float64
end
rand(x::Normal) = x.μ + rand()*x.σ
logprob(x::Normal, pt::Float64) = -(1.0/2.0)*( (pt-x.μ)/x.σ )^2 - (log(x.σ) + log(2*pi))

# Hierarchical addressing with a trie for use in trace.
struct Trie{K,V}
    leaf_nodes::Dict{K, V}
    internal_nodes::Dict{K, Trie{K, V}}
end
Trie{K,V}() where {K,V} = Trie(Dict{K,V}(), Dict{K,Trie{K,V}}())
Base.println(trie::Trie) = JSON.print(trie, 4)

function insert_leaf!(trie::Trie, addr, value)
    trie.leaf_nodes[addr] = value
end

# For now, the AddressMap (which uses the trie structure) has just Float64 values.
const AddressMap = Trie{Symbol, Float64}
const DependencyGraph = MetaGraph

# Trace to track score and dependency graph.
mutable struct Trace
    # Initialized at construction.
    dependencies::DependencyGraph
    address_map::AddressMap
    score::Float64

    function Trace()
        dependencies = MetaGraph()
        set_indexing_prop!(dependencies, :name)
        trie = AddressMap()
        new(dependencies, trie, 0.0)
    end
end
insert_vertex!(addr::Variable, g::MetaGraph) = !(addr in [get_prop(g, i, :name) for i in vertices(g)]) && add_vertex!(g, :name, addr)
insert_edge!(par::Variable, ch::Variable, g::MetaGraph) = add_edge!(g, g[par, :name], g[ch, :name])

# Here begins the IR analysis section...
function (t::Trace)(::typeof(rand), a::T) where {T <: Randomness}
    result = rand(a)
    label = gensym() # replace with addressing
    insert_leaf!(t.address_map, label, result);
    t.score += logprob(a, result)
    result
end

# Applying an instance of MetaGraph stores all the dependencies in the graph.
function (g::MetaGraph)(ir)
    ir == nothing && return
    blocks = IRTools.blocks(ir)
    for block in blocks

        # This handles IR statements
        for (x, st) in block
            l = st.expr.args[1]
            parents = filter(x -> x isa Variable && x in keys(ir) , st.expr.args)
            for par in parents
                par_st = ir[par]
                par_l = par_st.expr.args[1]
                par_l_type = typeof(eval(par_l))
                if (par_l isa GlobalRef && (par_l_type == DataType && supertype(eval(par_l)) == Randomness) || par_l_type == typeof(rand))
                    insert_vertex!(par, g)
                    insert_vertex!(x, g)
                    insert_edge!(par, x, g)
                end
            end
        end

        # This handles dependencies implied by branching on randomness
        for succ_block in IRTools.successors(block)
            branches = IRTools.branches(block, succ_block)
            pars = collect(Iterators.flatten(map(y -> filter(x -> x isa Variable, IRTools.arguments(y)), branches)))
            map(x -> insert_vertex!(x, g), pars)
            ch = filter(x -> x isa Variable, IRTools.arguments(succ_block))
            map(x -> insert_vertex!(x, g), ch)
            for i in pars
                for child in ch
                    insert_edge!(i, child, g)
                end
            end
        end
    end
end

# Multiple dispatch allows the recurse call in the Trace dynamo to easily implement probabilistic tracing.
function (t::Trace)(::typeof(rand), a::T) where {T <: Randomness}
    result = rand(a)
    label = gensym() # replace with addressing
    insert_leaf!(t.address_map, label, result);
    t.score += logprob(a, result)
    result
end

# This dynamo transforms the function into a probabilistic interpretation.
@dynamo function (t::Trace)(a...)
    ir = IR(a...)
    ir == nothing && return
    recurse!(ir)
    return ir
end

# Automatically constructs the IR dependency graph and then produces the trace.
macro probabilistic(fn, args)
    return quote
        ir = @code_ir $fn($args...)
        tr = Trace()

        # Construct dependency graph.
        tr.dependencies(ir)

        # This uses the IR pass.
        result = tr($fn, $args...)
        (result, tr)
    end
end

# Here's how you use this infrastructure.
function hierarchical_disgust(z::Float64)

    # These are equivalent to 'traced' randomness sources.
    x = rand(Normal(z, 1.0))
    y = rand(Normal(x, 1.0))
    m = rand(Normal(y, 1.0)) + 5.0
    if 0 < x
        n = rand(Normal(m, 1.0))
    else
        n = 0
    end
    q = 0

    # This is 'untraced' but still can be caught by the static pass.
    while rand(Normal(n, 10.0)) < 20.0
        q += 1
    end

    # Allocate an array and fill it with randomness!
    p = []
    push!(p, rand(Normal(0.0, 1.0)))
    for i in 2:Int(floor(rand(Normal(100.0, 5.0))))
        push!(p, rand(Normal(p[i-1], 1.0)))
    end
    return p
end

ir = @code_ir hierarchical_disgust(5.0)
println(ir)

result, trace = @probabilistic(hierarchical_disgust, (5.0, ))
println(trace.address_map)
labels = [props(trace.dependencies, i)[:name] for i in vertices(trace.dependencies)]

draw(PDF("graphs/dependency_graph.pdf", 16cm, 16cm), gplot(trace.dependencies, nodelabel = labels, arrowlengthfrac = 0.1, layout=circular_layout))
end # module
