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

@testset "Contextual execution" begin
    @testset "Unconstrained generate" begin
    end

    @testset "Constrained generate" begin
    end
    
    @testset "Update" begin
    end

    @testset "Regenerate" begin
    end

    @testset "Propose" begin
    end
    
    @testset "Score" begin
    end
end

