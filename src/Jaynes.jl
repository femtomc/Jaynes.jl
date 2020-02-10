module Jaynes

# Meta-programming required :)
using MacroTools
using IRTools: @dynamo, IR, xcall, self, recurse!

# Source of randomness.
abstract type Randomness end
struct Normal <: Randomness
    μ::Float64
    σ::Float64
end
rand(x::Normal) = μ + rand()*σ

# Hierarchical addressing for use in trace.
struct Trie{K,V}
    leaf_nodes::Dict{K, V}
    internal_nodes::Dict{K, Trie{K, V}}
end
insert!(trie::Trie{K, V}, key::K, val::V) where {K, V} = 

const AddressMap = Trie{Symbol, Randomness}

# Trace to track score.
mutable struct Trace
    # Initialized by constructor.
    fn::Function
    args::Tuple
    address_map::AddressMap
    score::Float64

    # Not known at construction.
    retval::Any
    function Trace(fn, args)
        trie = AddressMap()
        new(fn, args, trie, 0.0)
    end
end
insert!(tr::Trace, x::Symbol, lw::Float64) = 

# Here begins the IR analysis section...
function probabilistic(::typeof(rand), a::T) where {T <: Randomness} = rand(a)

# This dynamo analyzes the IR, transforming the function into a probabilistic interpretation.
@dynamo function probabilistic(a...)
    ir = IR(a...)
    ir == nothing && return
end

@macro function probabilistic(fn, args...)
    return quote
        trace = Trace(fn, args)
        probabilistic() do
            fn(args)
        end
    end
end

# Here's how you use this infrastructure.
function hierarchical_normal()
    x = rand(Normal(0.0, 1.0))
    y = rand(Normal(x, 1.0))
    return y
end

trace, weight = @probabilistic hierarchical_normal()

end # module
