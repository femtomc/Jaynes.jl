module LogPDFTransform

using Jaynes

using IRTools
using IRTools: func

using Distributions
using DistributionsAD: logpdf

using Zygote: gradient

# Stochastic kung-foo!
function foo(z::Float64)
    x = rand(Normal(z, 1.0))
    y = rand(Normal(x, 1.0))
    l = rand(Normal(x + y, 3.0))
    return l
end

ir = @code_ir foo(5.0)
println("\nOriginal:\n", ir, "\n")

# Testing.
transformed = @code_ir logpdf_transform! foo(5.0)
println("Transformed:\n", transformed, "\n")
logprob = func(transformed)

println(logprob(0.3, 3.0, 5.0, 5.0))
for i in [1.0, 2.0, 5.0]
    println(logprob(i, -i, i, 5.0))
end

grad = gradient((x, y, z, k) -> logprob(x, y, z, k), 0.3, 3.0, 5.0, 5.0)
println("\nGradient:\n", grad)

# Multi-variate stuff
function foo2()
    y = rand(Normal(0, 1))
    z = rand(Normal(0, 1))
    x = rand(MvNormal([y, z], [1.0 0.0; 0.0 1.0]))
    return x
end

ir = @code_ir foo2()
println("\nOriginal:\n", ir, "\n")

transformed = @code_ir logpdf_transform! foo2()
println("Transformed:\n", transformed, "\n")
logprob = func(transformed)
println(logprob(0.3, 0.3, [0.3, 3.0]))
grad = gradient((x, y, z) -> logprob(x, y, z), 0.3, 0.3, [0.3, 3.0])
println("\nGradient:\n", grad)

end #module
