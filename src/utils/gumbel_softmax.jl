function ∂Cat(probs::Vector{Float64}, τ)
    G = [rand(Gumbel(0.0, 1.0)) for i in 1 : length(probs)]
    lw = log.(probs)
    y = ( G + lw ) / τ
    logZ = Jaynes.lse(y)
    sample = exp.(y .- logZ)
    sample
end

# TODO: fix.
@primitive function logpdf(fn::typeof(∂Cat), probs, τ, sample::Vector{Float64})
    return sum(log.(probs .* sample))
end
