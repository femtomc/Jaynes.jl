module Profiling

include("../../src/Jaynes.jl")
using .Jaynes
using Distributions
using Profile
using PProf

function foo1()
    x = rand(:x, Normal, (3.0, 1.0))
    y = rand(:y, Normal, (x, 1.0))
    return y
end

function foo2()
    x = rand(:x, Normal, (3.0, 1.0))
    return x
end

obs = constraints([(:y, 1.0)])
Jaynes.importance_sampling(foo1, (), foo2, (), obs, 1)
Profile.clear_malloc_data()
trs, lnw, lmle = @profile Jaynes.importance_sampling(foo1, (), foo2, (), obs, 30000)
pprof()

end # module
