module GraphIR

# IRTools pass which extracts out primitive distributions from a method invocation. Fails if there is:
# 1. Mutation.
# 2. Iteration.

using IRTools
using IRTools: Variable, meta, IR
using Distributions
using MacroTools

abstract type Node end

struct DistributionSite{T} <: Node
    distribution::T
end

struct CallSite <: Node
    func::Function
end

struct DependencyGraph
    nodes::Dict{Symbol, Node}
    addresses::Vector{Symbol}
    adjacencies::Dict{Symbol, Symbol}
    DependencyGraph() = new(Dict{Symbol, Node}(), Vector{Symbol}(), Dict{Symbol, Symbol}())
end

mutable struct GraphContext
    associated::Dict{Variable, Symbol}
    dependency::Dict{Variable, Variable}
    graph::DependencyGraph
    func::Function
    GraphContext() = new(Dict{Variable, Symbol}(), Dict{Variable, Variable}(), DependencyGraph())
end

function graph_pass!(ctx::GraphContext, ir)
    for (v, st) in ir
        in_dict = v
        expr = st.expr
        MacroTools.postwalk(expr) do k
            k in keys(ctx.associated) && begin
                in_dict = k
                ctx.associated[v] = ctx.associated[k]
            end
            k
        end
        ctx.dependency[v] = in_dict
        expr.head == :call && expr.args[1] isa GlobalRef && begin
            ref = expr.args[1]
            if ref.name == :rand
                expr.args[2] isa QuoteNode && begin
                    ctx.associated[v] = expr.args[2].value
                    push!(ctx.graph.addresses, expr.args[2].value)
                end
                expr.args[3] isa GlobalRef && begin
                    val = Base.eval(@__MODULE__, expr.args[3])
                    if val isa Function
                        ctx.graph.nodes[expr.args[2].value] = CallSite(val)
                    elseif val <: Distribution
                        ctx.graph.nodes[expr.args[2].value] = DistributionSite(val)
                    end
                end
            end
        end
    end
end

# ---- TEST ---- #

function bar()
    rand(:z, Normal, (0.0, 1.0))
end

function foo()
    x = rand(:x, Normal, (0.0, 1.0))
    y = rand(:y, Normal, (x, 1.0))
    rand(:bar, bar, ())
    return y
end

ctx = GraphContext()
ir = IR(meta(Tuple{typeof(foo)}))
graph_pass!(ctx, ir)
println(ir)
println(ctx)

end # module
