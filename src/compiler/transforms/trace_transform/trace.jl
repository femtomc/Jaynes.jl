function var_ssa_map(f::Function, args...)
    lowered = @code_lowered f(args...)
    ir = @code_ir f(args...)

    # Parse IR for distributions - TODO: extraneous, keep around if useful.
    d = Dict{Variable, Any}()
    ks = Array{Variable, 1}([])
    for (v, st) in ir
        x = st.expr.args[1]
        if x isa GlobalRef && x.name in dists
            push!(ks, v)
            d[v] = x
        end
    end
    ks = union(arguments(ir), ks)
    slotnames = lowered.slotnames
    return d, Dict{Variable, Symbol}(zip(ks, slotnames)), slotnames[length(arguments(ir)) + 1:length(slotnames)]
end

# Note - currently copied from Gen.jl

mutable struct Trace
    call::Function
    trie::Trie{Symbol,ChoiceRecord}
    score::Float64
    args::Tuple
    static::Any
    addresses::Any
    function Trace(call::Function, args...)
        trie = Trie{Symbol,ChoiceRecord}()
        addresses = var_ssa_map(call, args...)[3]
        new(call, trie, 0.0, args, copy(addresses), reverse!(addresses))
    end
end

function Base.println(tr::Trace)
    println("------------ trace -----------")
    println("| Call: $(tr.call)")
    println("| Args: $(tr.args)")
    println("| Choice map:")
    map(x -> println("| ", x => tr.trie[x].val, " => Score: $(tr.trie[x].score)"), tr.static)
    println("| Distributions:")
    map(x -> println("| ", x => tr.trie[x].dist), tr.static)
    println("| Score: $(tr.score)")
    println("------------------------------")
end

function Base.println(tr::Trace, int::Int64)
    println("------------ trace $(int) ---------")
    println("|Call: $(tr.call)")
    println("|Args: $(tr.args)")
    println("|Choice map:")
    map(x -> println("|", x => tr.trie[x].val), tr.static)
    println("|Distributions:")
    map(x -> println("|", x => tr.trie[x].dist), tr.static)
    println("|Score: $(tr.score)")
    println("------------------------------")
end

get_choices(tr::Trace) = union(tr.args, map(x -> tr.trie[x].val, tr.static))
addresses(tr::Trace) = tr.addresses

function add_choice!(tr::Trace, addr::Symbol, dist, val, score::Float64)
    if haskey(tr.trie, addr)
        error("Value or subtrace already present at address $addr.
              Mutation is not allowed.")
    end
    tr.trie[addr] = ChoiceRecord(val, score, dist)
    tr.score += score
end

