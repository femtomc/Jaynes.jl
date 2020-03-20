# Note - currently copied from Gen.jl

abstract type ChoiceMap end

struct Trie{K, V} <: ChoiceMap
    leaf_nodes::Dict{K, V}
    internal_nodes::Dict{K, Trie{K, V}}
end

# Default.
Trie{K,V}() where {K,V} = Trie(Dict{K,V}(), Dict{K,Trie{K,V}}())

# Access.
leaf_nodes(trie::Trie) = trie.leaf_nodes
internal_nodes(trie::Trie) = trie.internal_nodes

function Base.values(trie::Trie)
    iterators = convert(Vector{Any}, collect(map(values, values(trie.internal_nodes))))
    push!(iterators, values(trie.leaf_nodes))
    Iterators.flatten(iterators)
end

function has_internal_node(trie::Trie, addr)
    haskey(trie.internal_nodes, addr)
end

function has_internal_node(trie::Trie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        has_internal_node(trie.internal_nodes[first], rest)
    else
        false
    end
end

function get_internal_node(trie::Trie, addr)
    trie.internal_nodes[addr]
end

function get_internal_node(trie::Trie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        get_internal_node(trie.internal_nodes[first], rest)
    else
        throw(KeyError(trie, addr))
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

function delete_internal_node!(trie::Trie, addr)
    delete!(trie.internal_nodes, addr)
    return isempty(trie.leaf_nodes) && isempty(trie.internal_nodes)
end

function delete_internal_node!(trie::Trie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        node = trie.internal_nodes[first]
        if delete_internal_node!(node, rest, new_node)
            delete!(trie.internal_nodes, first)
        end
    end
    return isempty(trie.leaf_nodes) && isempty(trie.internal_nodes)
end

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

function get_leaf_node(trie::Trie, addr)
    trie.leaf_nodes[addr]
end

function get_leaf_node(trie::Trie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        get_leaf_node(trie.internal_nodes[first], rest)
    else
        throw(KeyError(trie, addr))
    end
end

function set_leaf_node!(trie::Trie, addr, value)
    trie.leaf_nodes[addr] = value
end

function set_leaf_node!(trie::Trie{K,V}, addr::Pair, value) where {K,V}
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        node = trie.internal_nodes[first]
    else
        node = Trie{K,V}()
        trie.internal_nodes[first] = node
    end
    node = trie.internal_nodes[first]
    set_leaf_node!(node, rest, value)
end

function delete_leaf_node!(trie::Trie, addr)
    delete!(trie.leaf_nodes, addr)
    return isempty(trie.leaf_nodes) && isempty(trie.internal_nodes)
end

function delete_leaf_node!(trie::Trie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.internal_nodes[first])
        node = trie.internal_nodes[first]
        if delete_leaf_node!(node, rest)
            delete!(trie.internal_nodes, first)
        end
    end
    return isempty(trie.leaf_nodes) && isempty(trie.internal_nodes)
end

Base.setindex!(trie::Trie, value, addr) = set_leaf_node!(trie, addr, value)

function Base.merge!(a::Trie{K,V}, b::Trie{K,V}) where {K,V}
    merge!(a.leaf_nodes, b.leaf_nodes)
    for (key, a_sub) in a.sub
        if haskey(b.sub, key)
            b_sub = b.sub[key]
            merge!(a_sub, b_sub)
        end
    end
    for (key, b_sub) in b.sub
        if !haskey(a.sub, key)
            a.sub[key] = b_sub
        end
    end
    a
end

Base.isempty(trie::Trie) = isempty(trie.leaf_nodes) && isempty(trie.internal_nodes)
Base.haskey(trie::Trie, key) = has_leaf_node(trie, key)
Base.getindex(trie::Trie, key) = get_leaf_node(trie, key)
