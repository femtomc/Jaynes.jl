module SupportChecking

include("../src/Jaynes.jl")
using .Jaynes

#model1 = @jaynes function foo(x::Int)
#    x > 10 ? y ~ Normal(0.0, 1.0) : y ~ Bernoulli(0.5)
#end (check)
#
#model2 = @jaynes function bar(x::Int)
#    y ~ Normal(0.0, 1.0)
#    y ~ Bernoulli(0.5)
#end (check)


end # module
