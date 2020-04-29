module LogPDFGradientAccumulator

using Cassette
using Distributions
using DistributionsAD

Cassette.@context J;

# Equivalent to gradient tape.
mutable struct Trace
    result_tape::Array{Number, 1}
    logpdf_tape::Array{Function, 1}
    Trace() = new([], [])
end

# Trace with context.
function Cassette.overdub(ctx::J, call::typeof(rand), d::T) where T <: Distribution
    result = rand(d)
    lambda = ȳ -> logpdf(d, ȳ)
    push!(ctx.metadata.result_tape, result)
    push!(ctx.metadata.logpdf_tape, lambda)
    return result
end

# Stochastic kung-foo!
function foo1()
    θ = rand(Beta(2, 2))
    μ = rand(Normal(1.0, 0.1))
    z = rand(Normal(μ, θ))
    return z
end

tr = Trace()
Cassette.overdub(Cassette.disablehooks(J(metadata = tr)), foo1)
println(tr.result_tape)

end #module
