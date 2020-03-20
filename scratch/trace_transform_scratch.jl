module TraceTransform

include("../src/Jaynes.jl")
using .Jaynes

using InteractiveUtils: @code_lowered

using IRTools
using IRTools: func

using Distributions
using DistributionsAD: logpdf

using Zygote: gradient

function tester()
    x = rand(Normal(0.0, 1.0))
    return x
end

# Stochastic kung-foo!
function foo(z::Float64)
    x = rand(Normal(z, 6.0))
    y = rand(Normal(x, 1.0))
    l = rand(Normal(x + y, 3.0))
    return y
end

function foo2()
    θ = rand(Beta(2, 2))
    μ = rand(Normal(1.0, 0.0))
    z = rand(Normal(μ, θ))
    x = Array{Float64, 1}(undef, 50)
    for i in 1:50
        x[i] = rand(Normal(z, 0.5))
    end
    return x
end

tr = Trace(foo, 4.0)
tr() do
    foo(4.0)
end

println(tr)
println(get_choices(tr))

println(@code_ir foo2())
lowered = @code_lowered foo2()
println(lowered.slotnames)


end #module
