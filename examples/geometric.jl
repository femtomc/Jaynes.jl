module Geometric

include("../src/Walkman.jl")
using .Walkman
using Distributions

geo(p::Float64) = rand(:flip, Bernoulli, (p, )) == 1 ? 0 : 1 + rand(:geo, geo, p)

tr, weight = trace(geo, (0.3, ))
display(tr)

end # module
