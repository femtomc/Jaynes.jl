module Geometric

include("../../src/Jaynes.jl")
using .Jaynes
using Distributions

geo(p::Float64) = rand(:flip, Bernoulli(p)) == 1 ? 0 : 1 + rand(:geo, geo, p)
fn = () -> geo(0.4)
crazy = () -> (() -> fn())()
call = Jaynes.trace(crazy, ())
display(call.trace)

end # module
