module HiddenMarkovModel

include("../src/Jaynes.jl")
using .Jaynes
using Distributions

function kernel(prev_latent::Float64)
    z = rand(:z, Normal(prev_latent, 1.0))
    if z > 2.0
        x = rand(:x, Normal(z, 3.0))
    else
        x = rand(:x, Normal(z, 10.0))
    end
    return z
end

# Specialized Markov call site informs tracer of dependency information.
kernelize = n -> markov(:k, kernel, n, 5.0)

simulation = () -> begin
    ps = initialize_filter(selection(), 1000, kernelize, (1, ))
    for i in 2:100
        sel = selection((:k => i => :x, 5.0))

        # Complexity of filter step is constant as a size of the trace.
        @time filter_step!(sel, ps, NoChange(), (i,))
    end
    return ps
end

ps = simulation()

end # module
