module SupportChecking

include("../src/Jaynes.jl")
using .Jaynes

model1 = @jaynes function foo()
    5 > 10 ? y ~ Normal(0.0, 1.0) : y ~ Bernoulli(0.5)
end

model2 = @jaynes function bar(x::Int)
    y ~ Normal(0.0, 1.0)
    y ~ Normal(0.5, 1.0)
    x > 10 ? y ~ Normal(0.0, 1.0) : y ~ Normal(0.3, 3.0)
end (DefaultPipeline)

end # module
