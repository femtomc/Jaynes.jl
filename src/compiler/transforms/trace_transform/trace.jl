# Note - currently copied from Gen.jl

mutable struct Trace
    call::Function
    record_count::Int
    trie::Trie{Symbol, Tuple{Int, ChoiceRecord}}
    score::Float64
    args::Tuple
    retval::Any
    function Trace(call::Function, args...)
        trie = Trie{Symbol,Tuple{Int, ChoiceRecord}}()
        new(call, 0, trie, 0.0, args, nothing)
    end
end

function Base.println(tr::Trace)
    ks = collect(keys(tr.trie.leaf_nodes))
    println("------------ trace -----------")
    println("| Call: $(tr.call)")
    println("| Args: $(tr.args)")
    choices = map(x -> (x, tr.trie[x]), ks)
    sort!(choices, by = x -> x[2][1])
    println("|\n| Choice map:")
    map(x -> println("| ", x[1] => x[2][2].val, " => Score: $(x[2][2].score)"), choices)
    println("|\n| Distributions:")
    map(x -> println("| ", x[1] => x[2][2].dist), choices)
    println("|\n| Score: $(tr.score)")
    println("|\n| Return value type: $(typeof(tr.retval))")
    println("------------------------------")
end

get_choices(tr::Trace) = union(tr.args, map(x -> tr.trie[x].val, tr.static))

function record!(tr::Trace, addr::Symbol, dist, val, score::Float64)
    if haskey(tr.trie, addr)
        error("Value or subtrace already present at address $addr.
              Mutation is not allowed.")
    end
    tr.trie[addr] = (tr.record_count, ChoiceRecord(val, score, dist))
    tr.record_count += 1
    tr.score += score
end

