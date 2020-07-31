module Learnable

include("../src/Jaynes.jl")
using .Jaynes

function learnable_kernel(x::Float64)
    q = learnable(:q)
    m = rand(:m, Normal(x + q[1], exp(q[2])))
    return m
end

function learnable_sim(q::Float64, len::Int)
    z = learnable(:z)
    x = markov(:x, learnable_kernel, len, z + q)
    return x
end

# Initial parameters.
len = 200
ps = parameters([(:x, :q) => [1.0, 1.0],
                 (:z, ) => 3.0])
sel = selection([(:x, i, :m) => 20.0 for i in 1 : len])

# Train.
ps = train(sel, ps, learnable_sim, 0.0, len; 
           opt = ADAM(0.05, (0.9, 0.8)), 
           iters = 3000)
display(ps)

end # module
