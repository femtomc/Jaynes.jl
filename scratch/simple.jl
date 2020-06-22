module Effects

include("../src/Jaynes.jl")
using .Jaynes
using Distributions

geo(p::Float64) = rand(:flip, Bernoulli, p) == 1 ? 0 : 1 + rand(:geo, geo, p)
call = trace(geo, (0.5,))
println(call.trace)

end #module
