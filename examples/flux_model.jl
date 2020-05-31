module FluxModel

include("../src/Jaynes.jl")
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
    return out
end

function bar(ins::Vector{Float64})
    params = model(ins)
    y = rand(:y, Normal, params)
    return y
end

ctx, tr, score = trace(bar, [0.6, 1.0, 3.0, 5.0, 0.3])
display(tr)

end # module
