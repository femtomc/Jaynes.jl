module ProposalBenchmarks

include("../src/Jaynes.jl")
using .Jaynes
using Distributions
using Gen
using BenchmarkTools

function foo1()
    x = rand(:x, Normal, (3.0, 1.0))
    y = rand(:y, Normal, (x, 1.0))
    return y
end

function foo2()
    x = rand(:x, Normal, (3.0, 1.0))
    return x
end

@gen function foo3()
    x = @trace(normal(3.0, 1.0), :x)
    y = @trace(normal(x, 1.0), :y)
    return y
end

@gen function foo4()
    x = @trace(normal(3.0, 1.0), :x)
    return x
end

obs = constraints([(:y, 1.0)])
Jaynes.importance_sampling(foo1, (), obs, 1)
println("Jaynes - proposal sampling from prior. Num_samples = 30000.")
trs, lnw, lmle = @btime Jaynes.importance_sampling(foo1, (), obs, 30000)
println(lmle)
println(sum(lnw)/length(lnw))
println("Jaynes - proposal sampling from proposal. Num_samples = 30000.")
trs, lnw, lmle = @btime Jaynes.importance_sampling(foo1, (), foo2, (), obs, 30000)
println(lmle)
println(sum(lnw)/length(lnw))
obs = choicemap(:y => 1.0)
Gen.importance_sampling(foo3, (), obs, 1)
println("Gen - proposal sampling from prior. Num_samples = 30000.")
trs, lnw, lmle = @btime Gen.importance_sampling(foo3, (), obs, 30000)
println(lmle)
println(sum(lnw)/length(lnw))
println("Gen - proposal sampling from proposal. Num_samples = 30000.")
trs, lnw, lmle = @btime Gen.importance_sampling(foo3, (), obs, foo4, (), 30000)
println(lmle)
println(sum(lnw)/length(lnw))

end # module