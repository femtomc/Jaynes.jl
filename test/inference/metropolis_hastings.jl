function LinearGaussian(μ::Float64, σ::Float64)
    α = 5.0
    x = rand(:x, Normal(μ, σ))
    y = rand(:y, Normal(α * x, 1.0))
    return y
end

function LinearGaussianProposal()
    α = 10.0
    x = rand(:x, Normal(α * 3.0, 3.0))
end

@testset "Metropolis-Hastings" begin
    ret, cl = simulate(LinearGaussian, 0.0, 1.0)
    sel = Jaynes.target([(:x, )])

    @testset "Linear Gaussian model" begin
        obs = target([(:y, ) => 5.0])
        ret, cl, w = generate(obs, LinearGaussian, 0.0, 1.0)
        for i in 1 : 100
            new, discard = Jaynes.metropolis_hastings(sel, cl)
        end
    end

    @testset "Linear Gaussian proposal" begin
    end
end
