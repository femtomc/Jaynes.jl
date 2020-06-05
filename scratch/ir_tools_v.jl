module UseIR

using IRTools
using IRTools: @dynamo, IR, recurse!, xcall, self
using Distributions

Address = Union{Symbol, Pair}

struct Choice{T}
    val::T
    score::Float64
end

mutable struct Context
    tr::Dict{Symbol, Choice}
    stack::Vector{Address}
    Context() = new(Dict{Symbol, Any}(), Vector{Address}())
end
import Base.push!
Base.push!(ctx::Context, addr::Address) = push!(ctx.stack, addr)

function stack_interaction!(ir)
    for (v, st) in ir
        expr = st.expr
        if expr isa Expr && expr.head == :call
            if expr.args[1] isa GlobalRef && expr.args[1].name == :rand
                if expr.args[2] isa QuoteNode
                    insert!(ir, v, xcall(push!, self, expr.args[2]))
                end
            end
        end
    end
end

@dynamo function (ctx::Context)(a...)
    ir = IR(a...)
    ir == nothing && return
    stack_interaction!(ir)
    recurse!(ir)
    return ir
end

function (ctx::Context)(fn::typeof(rand), addr::Address, dist::Type, args)
    d = dist(args...)
    sample = rand(d)
    score = logpdf(d, sample)
    ctx.tr[addr] = Choice(sample, score)
    return sample
end

function (ctx::Context)(fn::typeof(rand), addr::Address, call::Function, args)
    ret = ctx() do
        call(args...)
    end
end

trace(fn, args) = begin
    ctx = Context()
    if !isempty(args)
        ctx() do
            fn(args)
        end
    else
        ctx() do
            fn()
        end
    end
    return ctx, ctx.tr
end

function trace(fn, args, num::Int)
    ctx = Context()
    trs = Vector{Dict{Address, Choice}}(undef, num)
    for i in 1:num
        if !isempty(args)
            ctx() do
                fn(args)
            end
            trs[i] = ctx.tr
        else
            ctx() do
                fn()
            end
            trs[i] = ctx.tr
        end
        ctx.tr = Dict{Address, Choice}()
    end
    return ctx, trs
end

# ----------- TEST ------------ #

function bar()
    y = rand(:y, Normal, (0.0, 1.0))
    return y
end

function foo()
    x = rand(:x, Normal, (0.0, 1.0))
    y = rand(:bar, bar, ())
    return x
end

ctx, tr = trace(foo, ())
println(ctx.stack)

end # module
