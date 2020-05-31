module Gradients

using Zygote
using Cassette
using Cassette: disablehooks, recurse
using InteractiveUtils: @code_lowered
using Distributions

import Base.rand
rand(addr::Symbol, d::Type, args) = rand(d(args...))
rand(addr::Symbol, d::Function, args) = d(args...)

Cassette.@context TapeContext

mutable struct GradientMeta
    tape::Dict{Symbol, NamedTuple}
    GradientMeta() = new(Dict{Symbol, Tuple}())
end

@inline function Cassette.overdub(ctx::TapeContext, fn::Function, args)
    println(fn)
    println(args)
    grad = Zygote.gradient(a -> fn(a...)[1], args)[1]
    println(grad)
    ctx.metadata.tape[Symbol(fn)] = (params = grad, )
    ret = recurse(ctx, fn, args...)
    return ret
end

@inline function Cassette.overdub(ctx::TapeContext,
                                  call::typeof(rand), 
                                  addr::Symbol, 
                                  dist::Type,
                                  args)
    d = dist(args...)
    sample = rand(d)
    grad = Zygote.gradient((args, a) -> logpdf(dist(args...), a), args, sample)
    ntuple = (params = grad[1], sample = grad[2])
    ctx.metadata.tape[addr] = ntuple
    return sample
end

@inline function Cassette.overdub(ctx::TapeContext,
                                  c::typeof(rand),
                                  addr::Symbol,
                                  call::Function,
                                  args)
    ret = recurse(ctx, call, args...)
    return ret
end

function bar(q::Float64)
    return rand(:m, Normal, (0.0, 1.0))
end

function foo(y::Float64, z::Float64)
    q = rand(:q, Normal, (0.0, 1.0))
    b = rand(:bar, bar, (q, ))
    return z + y + q
end

ctx = disablehooks(TapeContext(metadata = GradientMeta()))
ret = Cassette.overdub(ctx, foo, (5.0, 6.0))
println(ret)
println(ctx)
end # module
