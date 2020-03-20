module ClosureScratch

using InteractiveUtils: @code_lowered
using IRTools
using IRTools: IR, @dynamo, Variable, arguments, recurse!

using Distributions

using Jaynes: Trie, ChoiceRecord, dists

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


@dynamo function (tr::Trace)(m...)
    ir = IR(m...)
    ir == nothing && return
    recurse!(ir)
    return ir
end

function (tr::Trace)(call::typeof(rand), dist::T) where T <: Distribution
    result = call(dist)
    score = logpdf(dist, result)
    add_choice!(tr, pop!(tr.addresses), dist, result, score)
    return result
end


# Stochastic kung-foo!
function foo(z::Float64)
    x = rand(Normal(z, 6.0))
    y = rand(Normal(x, 1.0))
    l = rand(Normal(x + y, 3.0))
    return y
end

function foo2()
    θ = rand(Beta(2, 2))
    μ = rand(Normal(1.0, 0.0))
    z = rand(Normal(μ, θ))
    x = Array{Float64, 1}(undef, 50)
    for i in 1:50
        x[i] = rand(Normal(z, 0.5))
    end
    return x
end

tr = Trace(foo, 4.0)
tr() do
    foo(4.0)
end

end #module
