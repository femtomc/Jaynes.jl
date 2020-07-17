module Geometric

include("../src/Jaynes.jl")
using .Jaynes
using Distributions

geo(p::Float64) = rand(:flip, Bernoulli(p)) ? 1 : 1 + rand(:geo, geo, p)

# Example trace.
ret, cl = simulate(geo, 0.2)
display(cl.trace)

end # module
