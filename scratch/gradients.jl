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

@inline function Cassette.prehook(ctx::TapeContext, fn::Function, args...)
    grad = Zygote.gradient(a -> fn(a...)[1], args)[1]
    ctx.metadata.tape[Symbol(fn)] = (params = grad, )
    println(fn)
end

function foo(x::Float64)
    y = x + 10.0
    return y
end

ctx = TapeContext(metadata = GradientMeta())
Cassette.overdub(ctx, foo, 5.0)
println(ctx.metadata.tape)

end # module
