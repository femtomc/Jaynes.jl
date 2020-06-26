module JaynesTalkExample

include("../src/Jaynes.jl")
using .Jaynes
using Distributions

function wats_p()
    p = rand(:p, Beta(1, 1))
    q = rand(:q, Geometric(p))
    return q
end

function good_prop(q::Int)
    c = ceil(q)
    p = rand(:p, Beta(1, c))
end

obs = Jaynes.selection((:q, 100))
calls, lnw, lmle = Jaynes.importance_sampling(wats_p, (), good_prop, (50, ); observations = obs, num_samples = 50000)
println(lmle)
println(sum(map(calls) do c
                c[:p]
            end) / 50000)

end #module

