module InactivePathsExample

include("../src/Jaynes.jl")
using .Jaynes

model0 = @jaynes function bar(x::Float64)
    sleep(5.0)
end

model1 = @jaynes function foo()
    x ~ Normal(0.0, 1.0)
    y ~ Normal(x, 1.0)
    z ~ Normal(0.0, 1.0) 
    q ~ Normal(x, 1.0)
    m ~ bar(x)
end

# Actually see here that the logpdf has a dependency structure:
# x -> y 
# x -> q
# z
# x -> m
# which means that the choice at z is "inactive" with respect to the grad of logpdf(tr) for sample traces generated from this model.

tr = simulate(model1, ())
display(tr)

# If you want ∇ logpdf wrt z, is the AD smart enough to ignore bar?

end # module
