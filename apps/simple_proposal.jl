module SimpleProposal

include("../src/Jaynes.jl")
using .Jaynes
using Distributions

bar() = rand(:z, Normal(0.0, 1.0))

function foo()
    x = rand(:x, Normal(0.0, 1.0))
    y = x + rand(:y, Normal(0.0, 1.0))
    q = bar()
    return q + y
end

obs = constraints([(:foo => :y, 4.0)])
tr = Trace(obs)
tr() do
    foo()
end

println(obs)
println(tr, [:val])

end # module
