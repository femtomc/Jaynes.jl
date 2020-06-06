module Geometric

include("../../src/Jaynes.jl")
using .Jaynes
using Distributions

geo(p::Float64) = rand(:flip, Bernoulli, (p, )) == 1 ? 0 : 1 + rand(:geo, geo, p)
fn = () -> geo(0.4)
crazy = () -> (() -> fn())()
ctx, tr, weight = trace(crazy)
display(tr)

end # module
