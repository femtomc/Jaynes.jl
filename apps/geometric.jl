module SimpleProposal

include("../src/Walkman.jl")
using .Walkman
using Distributions

geo(p::Float64) = rand(:flip, Bernoulli, (p, )) == 1 ? 0 : 1 + rand(:geo, geo, p)

res, tr = trace(geo, 0.02)
println(tr, [:val])

end # module
