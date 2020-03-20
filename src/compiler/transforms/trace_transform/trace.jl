# Note - currently copied from Gen.jl

mutable struct Trace
    call::Function
    trie::Trie{Symbol,ChoiceRecord}
    score::Float64
    args::Tuple
    retval::Any
    function Trace(call::Function, args...)
        trie = Trie{Symbol,ChoiceRecord}()
        new(call, trie, 0.0, args, nothing)
    end
end

function Base.println(tr::Trace)
    ks = collect(keys(tr.trie.leaf_nodes))
    println("------------ trace -----------")
    println("| Call: $(tr.call)")
    println("| Args: $(tr.args)")
    println("|\n| Choice map:")
    map(x -> println("| ", x => tr.trie[x].val, " => Score: $(tr.trie[x].score)"), ks)
    println("|\n| Distributions:")
    map(x -> println("| ", x => tr.trie[x].dist), ks)
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
    tr.trie[addr] = ChoiceRecord(val, score, dist)
    tr.score += score
end

