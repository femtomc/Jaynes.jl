module Profiling

include("../src/Walkman.jl")
using .Walkman
using Distributions

function foo1()
    z = 10.0
    x = rand(:x, Normal, (z, 1.0))
    y = rand(:y, Normal, (x, 1.0))
    return y
end

obs = constraints([(:y, 5.0)])
tr = trace(foo1, (), obs)
println(tr, [:val])

end # module
