module AutomaticAddressing

include("../src/Jaynes.jl")
using .Jaynes
using Random
using IRTools
using IRTools: @dynamo, IR, meta, Pipe, finish, recurse!, self, blocks
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
    ref
end

@inline addr_viz(tcs::TransformedCallSite) = display(tcs.ref)

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

@inline check_prim(name) = name in names(Base, all = true) || name in names(Core, all = true) || name in names(Core.Intrinsics, all = true)

function insert_pass(fn_name, ir)
    counter = 1
    num_blocks = length(blocks(ir))
    pr = Pipe(ir)
    for (v, st) in pr
        expr = st.expr
        if isexpr(expr, :call) && 
            !check_prim(Jaynes.unwrap(expr.args[1]))
            name = expr.args[1]
            pr[v] = Expr(:call, 
                         GlobalRef(@__MODULE__, :rand), 
                         QuoteNode(Symbol(Jaynes.unwrap(name))),
                         expr.args[1 : end]...)
        elseif check_rand(expr)
            pr[v] = Expr(:call, 
                         GlobalRef(@__MODULE__, :rand), 
                         counter,
                         expr.args[2 : end]...)
            counter += 1
        end
    end
    ir = finish(pr)
    ir
end

function check_for_rand_calls(ir)
    for (v, st) in ir
        check_rand(st.expr) && return true
    end
    return false
end

# Modified recurse call which avoids Base and Core.
function recur(ir, to = self)
    pr = Pipe(ir)
    for (x, st) in pr
        isexpr(st.expr, :call) && begin
            ref = Jaynes.unwrap(st.expr.args[1])
            !(ref in Jaynes.whitelist) && check_prim(ref) && continue
            pr[x] = Expr(:call, to, st.expr.args...)
        end
    end
    finish(pr)
end

@dynamo function (mx::TransformationContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    check_for_rand_calls(ir) || return ir
    ir = insert_pass(a[1], ir)
    ir = recur(ir)
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

@inline function (ctx::TransformationContext)(c::typeof(rand),
                                              addr::T,
                                              call::Function,
                                              args...) where T <: Jaynes.Address
    Jaynes.visit!(ctx, addr)
    ret, cl = auto_simulate(call, args...)
    Jaynes.add_call!(ctx, addr, cl)
    return ret
end


function viz_transform(fn, args...)
    tt = typeof((fn, args...))
    ir = IR(meta(tt))
    ir = insert_pass(fn, ir)
    Jaynes.flow_analysis(ir)
end

function auto_simulate(fn::Function, args...)
    ctx = AutoAddressing(Jaynes.Trace(), Jaynes.Empty())
    ret = ctx(fn, args...)
    return ret, TransformedCallSite(ctx.tr, ctx.score, fn, args, ret, viz_transform(fn, args...))
end

# ------------ Examples ------------ #

baz = () -> begin
    x = rand(Normal(0.0, 1.0))
    y = rand(Normal(x, 1.0))
    y
end

bar = () -> begin
    x = rand(Normal(0.0, 1.0))
    y = rand(Normal(x, 1.0))
    baz()
end

foo = () -> begin
    if rand(Normal(3.0, 1.0)) > 3.0
        y = rand(Normal(0.0, 1.0))
    else
        y = rand(Normal(5.0, 1.0))
    end
    q = rand(Normal(y, 5.0))
    y + q + baz()
end

@time auto_simulate(foo)
@time ret, cl = auto_simulate(foo)
display(cl.trace)
#addr_viz(cl)

end # module
