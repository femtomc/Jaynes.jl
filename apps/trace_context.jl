module TraceContext

using Cassette
using Cassette: recurse

include("../src/Walkman.jl")
using .Walkman
using Distributions

# --------------------- TEST ---------------------- #

bar() = rand(:z, Normal, (3.0, 1.0))

function foo1()
    x = rand(:x, Normal, (3.0, 1.0))
    y = rand(:y, Normal, (x, 1.0))
    q = rand(:bar, bar, ())
    return y  + q
end

res, tr = trace(foo1, ())
println(tr, [:val])

end # module
