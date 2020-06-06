module FluxModel

include("../../src/Jaynes.jl")
using .Jaynes
using Distributions
using Flux

model = Chain(Dense(10, 5, σ),
              Dense(5, 2),
              softmax)

function baz(x::Float64)
    y = rand(:y, Normal, (0.0, 1.0))
    out = Vector{Float64}(undef, 10)
    for i in 1:10
        out[i] = rand(:q => i, Normal, (y, 5.0))
    end
    z = rand(:z, Normal, (sum(out), 1.0))
    return z
end

function bar(ins::Vector{Float64})
    params = model(ins)
    y = rand(:y, Normal, (0.0, 1.0))
    return y
end

obs = constraints([(:z, 5.0)])
ctx, trs, lnw, lmle = Jaynes.importance_sampling(baz, (5.0, ), bar, (ones(10), ), obs, 50000)
println(lmle)

end # module
