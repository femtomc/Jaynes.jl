module Simple

include("../src/Walkman.jl")
using .Walkman
using Distributions

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
Walkman.importance_sampling(foo1, (), obs, 1)
println("Walkman - proposal sampling from prior. Num_samples = 10000.")
trs, lnw, lmle = Walkman.importance_sampling(foo1, (), obs, 10000)
println(lmle)
println(trs[1], [:val])
println("--------------------------------------------------")
Walkman.importance_sampling(foo1, (), foo2, (), obs, 1)
println("Walkman - proposal sampling from proposal. Num_samples = 10000.")
trs, lnw, lmle = Walkman.importance_sampling(foo1, (), foo2, (), obs, 10000)
println(lmle)
end # module
