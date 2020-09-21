module HierarchicalModel

include("../src/Jaynes.jl")
using .Jaynes
using Gen

jmodel = @jaynes function flipper(p)
    burglary ~ Bernoulli(p)
    x ~ Bernoulli(burglary ? 0.75 : 0.01)
    x
end

@gen function generate_flipper()
    p ~ beta(2, 2)
    x ~ jmodel(p)
    x
end

# Inference.
tr = simulate(generate_flipper, ())
display(get_choices(tr))

end # module
