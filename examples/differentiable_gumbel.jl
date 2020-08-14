module DiffGumbel

include("../src/Jaynes.jl")
using .Jaynes

function ∂Cat(probs::Vector{Float64}, τ)
    G = [rand(Gumbel(0.0, 1.0)) for i in 1 : length(probs)]
    lw = log.(probs)
    y = ( G + lw ) / τ
    logZ = Jaynes.lse(y)
    sample = exp.(y .- logZ)
    sample
end

@primitive function logpdf(fn::typeof(∂Cat), probs, τ, sample::Vector{Float64})
    return sum(log.(probs .* sample))
end

model = () -> begin
    cat = rand(:G, ∂Cat, [0.3, 0.7], 0.5)
    if findmax(cat)[2] == 2
        rand(:x, Normal(10.0, 3.0))
    else
        rand(:x, Normal(2.0, 3.0))
    end
end

ret, cl = simulate(model)
display(cl.trace)
tg = target([(:G, )])
_, _, gs = get_choice_gradients(tg, cl, nothing)
display(gs)

end # module
