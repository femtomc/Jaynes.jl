module GraphIRScratch

using Cassette
using IRTools
using IRTools: blocks
using Mjolnir
using Mjolnir: Basic, AType, Const, abstract
using Distributions
Address = Union{Symbol, Pair}

import Base.rand
rand(addr::T, d::Type, args) where T <: Address = rand(d(args...))
rand(addr::T, d::Function, args) where T <: Address = d(args...)
rand(addr::T, lit::K) where {T <: Address, K <: Union{Number, AbstractArray}} = lit

Mjolnir.@abstract Basic rand(addr::Symbol, d::Type, args) = Union{Int, Float64}
Mjolnir.@abstract Basic rand(addr::Address, d::Type, args) = Union{Int, Float64}

abstract type GraphIR end

struct ToplevelCallNode <: GraphIR
    addr::Address
    call::Function
    args::Tuple
end

struct CallNode <: GraphIR 
    addr::Address
    call::Function
    args::Tuple
end

struct DistributionNode <: GraphIR 
    addr::Address
    dist::Type
    args::Tuple
end

struct WavefolderNode <: GraphIR 
    addr::Address
    call::Function
    iters::Int
    args::Tuple
end

Cassette.@context GraphConstructionCtx
mutable struct GraphMeta
    toplevel::ToplevelCallNode
    prev_node::GraphIR
    nodes::Dict{Address, GraphIR}
    dependencies::Dict{Address, Address}
    GraphMeta(tn::ToplevelCallNode) = new(tn, tn, Dict{Address, GraphIR}(:top => tn), Dict{Address, Address}())
end

import Base.push!
function push!(gm::GraphMeta, node::GraphIR)
    pnode = gm.prev_node
    gm.nodes[node.addr] = node
    gm.dependencies[pnode.addr] = node.addr
    gm.prev_node = node
end

# Static pass which checks for mutation, iteration, or any branching.
function pure_pass(ir)
    for (v, st) in ir
        ex = st.expr
        ex isa Expr && ex.head == :call && ex.args[1] isa GlobalRef && begin
            callname = ex.args[1].name
            callname == :push! && error("GraphDSL: mutation is not supported.")
            callname == :setfield! && error("GraphDSL: mutation is not supported.")
            callname == :setindex! && error("GraphDSL: mutation is not supported.")
        end
    end
    !(length(blocks(ir)) == 1) && error("GraphDSL: control flow is not supported.")
end

function compile(fn::Function, args::Tuple)
    ir = Mjolnir.@trace fn(args...)
    pure_pass(ir)
    func = IRTools.func(ir)
    tn = ToplevelCallNode(:top, func, args)
    gm = GraphMeta(tn)
    ctx = GraphConstructionCtx(metadata = gm)
    Cassette.overdub(ctx, fn, args...)
    return ctx.metadata
end

@inline function Cassette.overdub(ctx::GraphConstructionCtx, 
                                  call::Function, 
                                  args::Tuple)
    cn = CallNode(gensym(), call, args)
    push!(ctx.metadata, cn)
    ret = call(args)
    return ret
end

@inline function Cassette.overdub(ctx::GraphConstructionCtx,
                                  call::typeof(rand), 
                                  addr::T, 
                                  dist::Type,
                                  args) where T <: Address
    dn = DistributionNode(addr, dist, args)
    push!(ctx.metadata, dn)
    return rand(dist(args...))
end

@inline function Cassette.overdub(ctx::GraphConstructionCtx,
                                  c::typeof(rand),
                                  addr::T,
                                  call::Function,
                                  args) where T <: Address
    cn = CallNode(addr, call, args)
    push!(ctx.metadata, cn)
    return call(args...)
end

# --------- TEST --------- #

function bar(z::Float64)
    return rand(:z, Normal, (z, 5.0))
end

function foo(x::Int)
    z = rand(:z, Normal, (0.0, 1.0))
    y = 0.0
    for i in 1:10
        y = rand(:y => i, Normal, (y, 1.0))
    end
    q = rand(:q, Normal, (y, 3.0))
    return rand(:bar, bar, (q, ))
end

graph = compile(foo, (7, ))
println(graph)

end # module
