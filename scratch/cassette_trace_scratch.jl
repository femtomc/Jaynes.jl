module CassetteTrace

using IRTools
using IRTools: recurse!, @dynamo, IR, func

using MacroTools
using MacroTools: postwalk

using Core: CodeInfo, SlotNumber, SSAValue
using Cassette
using Distributions
using InteractiveUtils

include("../src/Jaynes.jl")
using .Jaynes: Trie, ChoiceRecord, add_choice!

# Trace context.
Cassette.@context TraceCtx;

mutable struct Trace
    trie::Trie{Symbol,ChoiceRecord}
    score::Float64
    slotnames::Array{Symbol, 1}
    function Trace()
        trie = Trie{Symbol,ChoiceRecord}()
        new(trie, 0.0, Array{Symbol, 1}([]), Array{Symbol, 1}([]))
    end
end

function record!(tr::Trace, addr::Symbol, dist, val, score::Float64)
    if haskey(tr.trie, addr)
        error("Value or subtrace already present at address $addr.
              Mutation is not allowed.")
    end
    tr.trie[addr] = ChoiceRecord(val, score, dist)
    tr.score += score
end

function Base.println(tr::Trace)
    ks = collect(keys(tr.trie.leaf_nodes))
    println("------------ trace -----------")
    println("| Choice map:")
    map(x -> println("| ", x => tr.trie[x].val, " => Score: $(tr.trie[x].score)"), ks)
    println("|\n| Distributions:")
    map(x -> println("| ", x => tr.trie[x].dist), ks)
    println("| Score: $(tr.score)")
    println("------------------------------")
end

# Overdub.
function Cassette.overdub(ctx::TraceCtx, 
                 call::typeof(rand), addr::String, d::T) where T <: Distribution
    result = call(d)
    score = logpdf(d, result)
    println(addr)
    record!(ctx.metadata, Symbol(addr), d, result, score)
    return result
end

function Cassette.overdub(ctx::TraceCtx, 
                 call::typeof(rand), d::T) where T <: Distribution
    result = call(d)
    score = logpdf(d, result)
    record!(ctx.metadata, gensym(), d, result, score)
    return result
end

# -- Compiler pass -- #

# Utility function for inserting arguments into function calls in lowered code.
insert_addr_in_call = (expr, addr) -> postwalk(x -> @capture(x, f_(xs__)) ? (f isa GlobalRef && f.name == :rand ? :($f($addr, $(xs...))) : x) : x, expr)

function transform_sample_statements(ci::Core.CodeInfo)
    lowered = copy(ci)
    code = lowered.code

    # Mapping the code...
    identifier_dict = Dict(map(x -> (Core.SlotNumber(x[1]) => x[2]), enumerate(lowered.slotnames)))
    SSA_dict = Dict(map(x -> (Core.SSAValue(x[1]) => x[2]), enumerate(code)))

    # Insertion.
    transformed = map(line -> (line isa Expr && line.args[1] isa Core.SlotNumber) ? insert_addr_in_call(line, String(identifier_dict[line.args[1]])) : line, code)

    # Dependency insertion.
    #transformed = map(expr -> postwalk(y -> (y isa Core.SSAValue && @capture(SSA_dict[y], f_(xs__)) && (f isa GlobalRef && f.name == :rand)) ? insert_addr_in_call(transformed[y.id], y) : y, expr), transformed)

    # Insert new code.
    lowered.code = transformed
    return lowered
end

function insert_semantic_identifiers(::Type{<:TraceCtx}, reflection::Cassette.Reflection)
    lowered = reflection.code_info
    transformed = transform_sample_statements(lowered)
    return transformed
end

const insert_semantic_identifiers_pass = Cassette.@pass insert_semantic_identifiers 
# -- END COMPILER PASS --

# Tests.
function bar(x::Float64)
    z = rand(Normal(x, 10.0))
    return z
end

function foo(x::Float64)
    l = 0
    y = rand(Normal(x, 1.0))
    q = rand(Normal(y, 5.0))
    z = x + rand(Normal(y, 5.0))
    while rand(Normal(10, 10)) < 20
        z += rand(Normal(z, 0.5))
    end
    z = z + bar(z)
    return z
end

tr = Trace()
ir = @code_ir foo(5.0)

# Testing bar...
tr = Trace()
Cassette.overdub(Cassette.disablehooks(TraceCtx(pass = insert_semantic_identifiers_pass, metadata = tr)), bar, 5.0)
println(tr)

# Testing foo...
overdubbed_no_pass_ir = @code_ir Cassette.overdub(Cassette.disablehooks(TraceCtx(metadata = tr)), foo, 5.0)
overdubbed_ir = @code_ir Cassette.overdub(Cassette.disablehooks(TraceCtx(pass = insert_semantic_identifiers_pass, metadata = tr)), foo, 5.0)

# No pass execution.
tr = Trace()
println("No pass overdubbed:\n$(overdubbed_no_pass_ir)\n")
Cassette.overdub(Cassette.disablehooks(TraceCtx(metadata = tr)), foo, 5.0)
println(tr)

# Pass execution.
tr = Trace()
println("Overdubbed:\n$(overdubbed_ir)\n")
Cassette.overdub(Cassette.disablehooks(TraceCtx(pass = insert_semantic_identifiers_pass, metadata = tr)), foo, 5.0)
println(tr)

end #module
