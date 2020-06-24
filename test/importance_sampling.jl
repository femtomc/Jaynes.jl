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

@testset "Importance sampling" begin
    observations = Jaynes.selection((:y, 3.0))

    @testset "Linear Gaussian model" begin
        calls, lnw, lmle = Jaynes.importance_sampling(LinearGaussian, (0.0, 1.0), observations, 50000)
    end

    @testset "Linear Gaussian proposal" begin
        calls, lnw, lmle = Jaynes.importance_sampling(LinearGaussian, (0.0, 1.0), LinearGaussianProposal, (), observations, 50000)
    end
end

