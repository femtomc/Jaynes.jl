module TraceAnonymous

include("../src/Walkman.jl")
using .Walkman
using Distributions

geo(p::Float64) = rand(:flip, Bernoulli, (p, )) == 1 ? 0 : 1 + rand(:geo, geo, p)

tr = trace(p -> geo(p), (0.2, ))
println(tr, [:val])

end # module

