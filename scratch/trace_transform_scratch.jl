module TraceTransform

include("../src/Jaynes.jl")
using .Jaynes

using InteractiveUtils: @code_lowered

using IRTools
using IRTools: func

using Distributions

using Cassette

# Stochastic kung-foo!
function foo1()
    θ = rand(Beta(2, 2))
    μ = rand(Normal(1.0, 0.1))
    z = rand(Normal(μ, θ))
    return z
end

function foo2()
    θ = rand(Beta(2, 2))
    μ = rand(Normal(1.0, 0.1))
    z = rand(Normal(μ, θ))
    x = Array{Float64, 1}(undef, 50)
    for i in 1:10
        x[i] = rand(Normal(z, 0.5))
    end
    return x
end

# Recursion.
function foo3(z::Float64)
    if rand(Normal(0.0, 1.0)) > 1.0
        return z
    else
        return z + foo3(1.0)
    end
end

# Diamond.
function foo4(z::Float64)
    x = rand(Beta(2, 2))
    y = rand(Beta(2, 2))
    q = rand(Normal(z, x + y))
    return q
end

tr = @trace foo1
println(tr)

tr = @trace foo2
println(tr)

tr = @trace foo3 5.0
println(tr)

tr = @trace foo4 5.0
println(tr)

end #module
