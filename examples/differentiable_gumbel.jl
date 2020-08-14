module DiffGumbel

include("../src/Jaynes.jl")
using .Jaynes

function ∂Cat(probs::Vector{Float64}, τ)
    G = [rand(:G => i, Gumbel(0.0, 1.0)) for i in 1 : length(probs)]
    lw = log.(probs)
    y = ( G + lw ) / τ
    logZ = Jaynes.lse(y)
    sample = exp.(y .- logZ)
    sample
end

cross_entropy(p1::Vector{Float64}, p2::Vector{Float64}) = sum(-log.(p1) .* p2)

model = probs -> begin
    sample = rand(:SFG, ∂Cat, [0.3, 0.7], 0.5)
    factor(cross_entropy(sample, probs))
end

test = () -> begin
    ret, cl = simulate(model, [1.0, 0.0])

    tg = target([(:SFG, :G => 1),
                 (:SFG, :G => 2)])
    calls = []
    for i in 1 : 1000
        @time cl, _ = hmc(tg, cl)
        i > 500 && i % 30 == 0 && begin
            push!(calls, cl)
            display(cl.trace)
        end
    end
end

test()

end # module
