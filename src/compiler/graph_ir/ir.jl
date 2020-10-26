# ------------ IR vocab ------------ #

abstract type GraphNode end

mutable struct FunctionArgumentNode <: GraphNode
    var::Variable
    type
    FunctionArgumentNode(v::Variable) = new(v)
end

struct InlinedConstantNode{V} <: GraphNode
    v::V
end

struct ConstantNode{V} <: GraphNode
    var::Variable
    v::V
end

struct DistributionNode{N, D <: GlobalRef} <: GraphNode
    var::Variable
    args::NTuple{N, GraphNode}
    dist::D
end

struct JuliaCallNode{N, F <: GlobalRef} <: GraphNode
    var::Variable
    args::NTuple{N, GraphNode}
    fn::F
end

struct TraceNode{N} <: GraphNode
    var::Variable
    addr::Symbol
    args::NTuple{N, GraphNode}
end

struct ReturnNode{N} <: GraphNode
    var::Variable
    args::NTuple{N, GraphNode}
end

struct GraphIR
    map::Dict{Variable, GraphNode}
    nodes::Vector{GraphNode}
    parents::Dict{Variable, Set{Variable}}
    children::Dict{Variable, Set{Variable}}
    GraphIR() = new(Dict{Variable, GraphNode}(),
                    GraphNode[], 
                    Dict{Symbol, Vector{Symbol}}(),
                    Dict{Symbol, Vector{Symbol}}())
    GraphIR(ancestors, reach) = new(Dict{Variable, GraphNode}(),
                                   GraphNode[], 
                                   ancestors,
                                   reach)
end

function Base.display(graph_ir::GraphIR)
    println(" __________________________________\n")
    println("              Graph IR\n")
    display(graph_ir.map)
    println("\n ------------ Children ------------\n")
    for n in graph_ir.nodes
        haskey(graph_ir.children, n.var) ? println(" $(n.var) => $(graph_ir.children[n.var])") : println(" $(n.var)")
    end
    println("\n ------------ Ancestors ------------\n")
    for n in graph_ir.nodes
        haskey(graph_ir.parents, n.var) ? println(" $(n.var) => $(graph_ir.parents[n.var])") : println(" $(n.var)")
    end
    println(" __________________________________\n")
end


# ------------ Derive graph IR from SSA IR ------------ #

@inline interpret(graph_ir::GraphIR, a) = InlinedConstantNode(a)
@inline interpret(graph_ir::GraphIR, v::Variable) = getindex(graph_ir.map, v)

function interpret(graph_ir::GraphIR, v::Variable, st::IRTools.Statement)::GraphNode
    st.expr isa Expr || return ConstantNode(v, st.expr)
    st.expr.head == :call && begin
        unwrapped = Jaynes.unwrap(st.expr.args[1])
        if Symbol("Distributions.$unwrapped") in Jaynes.distributions
            return DistributionNode(v, tuple(map(st.expr.args[2 : end]) do a
                                              interpret(graph_ir, a)
                                          end...), st.expr.args[1])
        elseif unwrapped == :trace
            return TraceNode(v, Jaynes.unwrap(st.expr.args[2]), tuple(map(st.expr.args[3 : end]) do a
                                              interpret(graph_ir, a)
                                          end...))
        else
            return JuliaCallNode(v, tuple(map(st.expr.args[2 : end]) do a
                                              interpret(graph_ir, a)
                                          end...), st.expr.args[1])
        end
    end
end

function interpret!(graph_ir::GraphIR, 
                    v::IRTools.Variable)
    node = FunctionArgumentNode(v)
    push!(graph_ir.nodes, node)
    setindex!(graph_ir.map, node, v)
end

function interpret!(graph_ir::GraphIR, 
                    v::IRTools.Variable, 
                    st::IRTools.Statement,
                    flow::Jaynes.FlowAnalysis)
    node = interpret(graph_ir, v, st)
    push!(graph_ir.nodes, node)
    setindex!(graph_ir.map, node, v)
end

function graph_walk(ir::IRTools.IR, flow::Jaynes.FlowAnalysis)
    graph_ir = GraphIR(flow.ancestors, flow.reach)
    for v in arguments(ir)
        interpret!(graph_ir, v)
    end
    for (v, st) in ir
        interpret!(graph_ir, v, st, flow)
    end
    graph_ir
end

