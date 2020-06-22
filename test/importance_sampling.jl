function LinearGaussian(μ::Float64, σ::Float64)
    α = 5.0
    x = rand(:x, Normal(μ, σ))
    y = rand(:x, Normal(α * μ, 1.0))
    return y
end

@testset "Importance sampling" begin

    @testset "Linear Gaussian model" begin
        observations = Jaynes.selection([(:x, 3.0)])
        calls, lnw, lmle = Jaynes.importance_sampling(LinearGaussian, (0.0, 1.0), observations, 50000)
    end
end

