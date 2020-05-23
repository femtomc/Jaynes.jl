module GradientCore

using Cassette
using Cassette: recurse, Reflection, disablehooks, disablehooks, disablehooks, disablehooks
using Distributions
using DistributionsAD
using Zygote

Cassette.@context GradientContext

struct Accumulator
    tape::Dict{Symbol, Tuple}
end

function Cassette.overdub(::GradientContext, f::typeof(rand), addr::Symbol, dist::Type, args)
    sample = f(dist(args...))
    ∇ = gradient((x, args) -> logpdf(dist(args...), x), sample, args)
    println(∇)
    ctx.metadata.tape[addr] = ∇
    return sample
end
    

function foo1(z)
    x = rand(:x, Normal, (z, 1.0))
    y = rand(:y, Normal, (x, 1.0))
    return y
end

ctx = disablehooks(GradientContext(metadata = Accumulator(Dict{Symbol, Tuple}())))

out = Cassette.overdub(ctx, foo1, 5.0)
println(out)
println(ctx.metadata)

end # module
