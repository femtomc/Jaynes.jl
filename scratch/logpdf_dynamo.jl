module LogPDFTransform

using Jaynes

using Flux
using Flux: Recur, RNNCell
using CuArrays

using IRTools: func

using IRTools

using Distributions
using DistributionsAD: logpdf

# Stochastic kung-foo!
function foo(z::Float64)
    x = rand(Normal(z, 1.0))
    y = rand(Normal(x, 1.0))
    return y
end

# Transformed.
function logpdf_foo(z, x, y)
    x_logpdf = logpdf(Normal(z, 1), x)
    y_logpdf = logpdf(Normal(x, 1), y)
    return x_logpdf + y_logpdf
end

ir = @code_ir foo(5.0)
println(ir)

# Testing.
logprob = func(@code_ir logpdf_transform! foo(5.0))

println(logprob(0.3, 3.0, 5.0))

for i in [1.0, 2.0, 5.0]
    println(logprob(i, -i, i))
end

end #module
