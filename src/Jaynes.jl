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
using IRTools: @code_ir, @dynamo, IR, recurse!, Variable, isexpr
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

# Hierarchical addressing with a trie for use in trace. Implementation from Gen.jl.
struct Trie{K,V}
    leaf_nodes::Dict{K, V}
    internal_nodes::Dict{K, Trie{K, V}}
end
Trie{K,V}() where {K,V} = Trie(Dict{K,V}(), Dict{K,Trie{K,V}}())
Base.isempty(trie::Trie) = isempty(trie.leaf_nodes) && isempty(trie.internal_nodes)
Base.println(trie::Trie) = JSON.print(trie, 4)
Base.haskey(trie::Trie, key) = has_leaf_node(trie, key)

function has_leaf_node(trie::Trie, addr)
    haskey(trie.leaf_nodes, addr)
end

function has_leaf_node(trie::Trie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        has_leaf_node(trie.internal_nodes[first], rest)
    else
        false
    end
end

function set_internal_node!(trie::Trie{K,V}, addr, new_node::Trie{K,V}) where {K,V}
    if !isempty(new_node)
        trie.internal_nodes[addr] = new_node
    end
end

function set_internal_node!(trie::Trie{K,V}, addr::Pair, new_node::Trie{K,V}) where {K,V}
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        node = trie.internal_nodes[first]
    else
        node = Trie{K,V}()
        trie.internal_nodes[first] = node
    end
    set_internal_node!(node, rest, new_node)
end

function set_leaf!(trie::Trie, addr, value)
    trie.leaf_nodes[addr] = value
end

# For now, the AddressMap (which uses the trie structure) has just Float64 values.
const AddressMap = Trie{Symbol, Float64}

# DependencyGraph is just a MetaGraph.
const DependencyGraph = MetaGraph

#TODO: These insertion functions for the DependencyGraph are inefficient, because they check all vertices before inserting.
insert_vertex!(addr::Variable, g::DependencyGraph) = !(addr in [get_prop(g, i, :name) for i in vertices(g)]) && add_vertex!(g, :name, addr)
insert_edge!(par::Variable, ch::Variable, g::DependencyGraph) = add_edge!(g, g[par, :name], g[ch, :name])

# Trace to track score and dependency graph.
mutable struct Trace

    # Initialized at construction.
    dependencies::DependencyGraph
    address_map::AddressMap
    score::Float64

    function Trace()
        dependencies = DependencyGraph()
        set_indexing_prop!(dependencies, :name)
        trie = AddressMap()
        new(dependencies, trie, 0.0)
    end
end

##########################################
# Here begins the IR analysis section... #
##########################################

# Applying an instance of MetaGraph to the IR stores all the dependencies in the graph.
function (g::MetaGraph)(ir)
    ir == nothing && return
    blocks = IRTools.blocks(ir)
    for block in blocks

        # This handles IR statements
        for (x, st) in block
            l = st.expr.args[1]

            # Get all parents which are IR Variables
            parents = filter(x -> x isa Variable && x in keys(ir) , st.expr.args)
            for par in parents
                par_st = ir[par]
                par_l = par_st.expr.args[1]
                par_l_type = typeof(eval(par_l))

                # Check if it's a reference to randomness or a call to rand.
                if (par_l isa GlobalRef && (par_l_type == DataType && supertype(eval(par_l)) == Randomness) || par_l_type == typeof(rand))
                    insert_vertex!(par, g)
                    insert_vertex!(x, g)
                    insert_edge!(par, x, g)
                end
            end
        end

        # This handles dependencies implied by branching on randomness.
        # This has some edge cases which are missing in the analysis.
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
    # This needs to be re-worked properly. 
    # The problem is similar to what happened with Cassette tagging - you need 'non-local' information in this call but the call only knows about overwriting rand(a). 
    # Thus, we need to do some IR insertion to make this work right.

    result = rand(a)
    new_label = gensym()
    label = Pair(a.name, new_label) # replace with addressing
    new_map = AddressMap()
    set_leaf!(new_map, new_label, result)
    set_internal_node!(t.address_map, label, new_map)

    # Accumulate score.
    t.score += logprob(a, result)
    result
end

# This dynamo transforms the function into a probabilistic interpretation by 'overdubbing' (Cassette lingo) using the above defined dispatched method.
@dynamo function (t::Trace)(a...)
    ir = IR(a...)
    ir == nothing && return
    for (x, st) in ir
        println((st.line, st))
    end
    recurse!(ir) # recurse into calls in the IR and apply (t::Trace) there.
    return ir
end

# Convenience macro: automatically constructs the IR dependency graph and then produces the trace.
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

# Here's how you use this infrastructure on a particularly disgusting generative function...
function hierarchical_disgust(z::Float64)

    # These are equivalent to 'traced' randomness sources.
    x = rand(Normal(z, 1.0))
    y = rand(Normal(x, 1.0))
    m = rand(Normal(y, 1.0)) + 5.0

    # 'Simple' control flow from a probabilistic perspective.
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

    if y > 0.5
        return p
    else
        return q
    end
end
result, trace = @probabilistic(hierarchical_disgust, (5.0, ))

ir = @code_ir hierarchical_disgust(5.0)
labels = [props(trace.dependencies, i)[:name] for i in vertices(trace.dependencies)]
println(trace.address_map)

draw(PDF("graphs/dependency_graph.pdf", 16cm, 16cm), gplot(trace.dependencies, nodelabel = labels, arrowlengthfrac = 0.1, layout=circular_layout))

end # module
