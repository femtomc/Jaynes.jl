struct Trie{K, V} <: AddressMap{Value}
    internals::Dict{K, Trie{K, V}}
    leaves::Dict{K, V}
end
