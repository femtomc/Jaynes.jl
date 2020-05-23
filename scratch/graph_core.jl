module GraphCore

using IRTools
using IRTools: functional, CFG, interference, stackify, reloop
using Cassette
using Distributions

Cassette.@context GraphCtx

struct Argument
    id::Int
end

struct NodeBlock
    args_in::Tuple{Vararg{Argument}}
    model::Distribution
end

struct Graph
    adjacency::Dict{Symbol, Symbol}
    nodes::Vector{NodeBlock}
end

function foo1()
    x = rand(:x, Normal, (0.0, 1.0))
    y = rand(:y, Normal(x, 5.0))
    return y
end

function foo2()
    x = rand(:x, Normal, (0.0, 1.0))
    y = rand(:y, Normal, (x, 5.0))
    for i in 1:10
        println(i)
    end
    return y
end

function check_control_flow(ir)
    for (v, st) in ir
        expr = st.expr
        expr.args[1] isa GlobalRef && expr.args[1].name == :iterate && error("ControlFlowError: the static graph DSL does not support iteration.")
    end
end

function analyze(call, args)
    ir = @code_ir call(args...)
    check_control_flow(ir)
end

function analyze(call)
    ir = @code_ir call()
    check_control_flow(ir)
end

ir = @code_ir foo2()
println("IR:\n$(ir)\n")
#println("Functional:\n$(functional(ir))\n")
#println("CFG:\n$(CFG(ir))\n")
#println("Stackify:\n$(stackify(CFG(ir)))\n")
#println("Interference:\n$(interference(ir))")

end # module
