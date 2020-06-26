module Geometric

include("../../src/Jaynes.jl")
using .Jaynes
using Distributions

geo(p::Float64) = rand(:flip, Bernoulli(p)) == 1 ? 0 : 1 + rand(:geo, geo, p)
call = Jaynes.trace(geo, (0.2,))
display(call.trace)

end # module
