module SimpleProposal

include("../src/Jaynes.jl")
using .Jaynes
using Distributions

geo(p::Float64) = rand(:flip, Bernoulli(p)) == 1 ? 0 : 1 + geo(p)

tr = Trace()
tr() do
    geo(0.2)
end

println(tr, [:val])

end # module
