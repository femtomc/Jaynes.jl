module AutomaticAddressing

include("../src/Jaynes.jl")
using .Jaynes
using Random
using IRTools
using IRTools: @dynamo, IR, meta, Pipe, finish
using MacroTools

mutable struct TransformationContext{T <: Jaynes.AddressMap, 
                                     P <: Jaynes.AddressMap} <: Jaynes.ExecutionContext
    tr::T
    score::Float64
    visited::Jaynes.Visitor
    params::P
end

struct TransformedCallSite{J, K} <: Jaynes.CallSite
    trace::Jaynes.DynamicTrace
    score::Float64
    fn::Function
    args::J
    ret::K
    ir
end

@inline ir_viz(tcs::TransformedCallSite) = display(tcs.ir)

function AutoAddressing(tr, params)
    TransformationContext(tr,
                          0.0, 
                          Jaynes.Visitor(), 
                          params)
end

@inline function check_rand(expr)
    expr isa Expr &&
    expr.head == :call &&
    expr.args[1] isa GlobalRef &&
    expr.args[1].name == :rand &&
    !(expr.args[2] isa QuoteNode) && 
    !(expr.args[2] isa GlobalRef)
end

function insert_pass(ir)
    counter = 0
    pr = Pipe(ir)
    for (v, st) in pr
        expr = st.expr
        if check_rand(expr)
            pr[v] = Expr(:call, 
                         GlobalRef(@__MODULE__, :rand), 
                         QuoteNode(Symbol(counter)), 
                         expr.args[2 : end]...)
            counter += 1
        end
    end
    finish(pr)
end

@dynamo function (mx::TransformationContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    ir = insert_pass(ir)
    ir = Jaynes.recur(ir)
    ir
end

@inline function (ctx::TransformationContext)(call::typeof(rand), 
                                              addr::T, 
                                              d::Distribution{K}) where {T <: Jaynes.Address, K}
    Jaynes.visit!(ctx.visited, addr)
    s = rand(d)
    Jaynes.add_choice!(ctx, addr, logpdf(d, s), s)
    return s
end

function viz_transform(fn, args...)
    tt = typeof((fn, args...))
    ir = IR(meta(tt))
    ir = insert_pass(ir)
    ir
end

function auto_simulate(fn::Function, args...)
    ctx = AutoAddressing(Jaynes.Trace(), Jaynes.Empty())
    ret = ctx(fn, args...)
    return ret, TransformedCallSite(ctx.tr, ctx.score, fn, args, ret, viz_transform(fn, args...))
end

# ------------ Examples ------------ #

model = () -> begin
    x = rand(Normal(0.0, 1.0))
    y = rand(Normal(x, 1.0))
    y
end

ret, cl = auto_simulate(model)
display(cl.trace)
ir_viz(cl)

end # module
