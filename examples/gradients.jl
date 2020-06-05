module Gradients

include("../src/Jaynes.jl")
using .Jaynes
using Distributions
using Flux
using Plots

function foo1()
    # Literals are tracked as trainable.
    x = rand(:x, 10.0)
    m = rand(:m, 5.0)
    t = rand(:t, 7.0)

    # Rand calls on distributions also get tracked and the dependency graph is created.
    y = rand(:y, Normal, (x, 1.0))
    z = rand(:z, Normal, (t, 3.0))
    for i in 1:10
        q = rand(:q => i, Normal, (m, 1.0))
    end
    return z
end

function foo2()
    x = rand(:x, 1.0)
    y = rand(:y, Normal, (x, 1.0))
    z = rand(:z, Normal, (x + 10, 13.0))
    for i in 1:10
        q = rand(:q => i, Normal, (x, 1.0))
    end
    return z
end

ctx, trs, _, _ = importance_sampling(foo2, (), 10000)
trained_ctx, losses = train!(ADAM(), foo1, (), trs)
plt = plot(losses)
display(plt)

end # module
