function LinearGaussian(μ::Float64, σ::Float64)
    α = 5.0
    x = rand(:x, Normal(μ, σ))
    y = rand(:y, Normal(α * x, 1.0))
    z = rand(:z, Normal(y, 5.0))
    return z
end

function LinearGaussianProposal()
    α = 10.0
    x = rand(:x, Normal(α * 3.0, 3.0))
    y = rand(:y, Normal(0.0, 1.0))
end

function OneSiteProposal()
    x = rand(:x, Normal(0.0, 1.0))
end

@testset "Importance sampling" begin
    z = 3.0
    observations = Jaynes.target([(:z,) =>  z])
    n_traces = 5

    @testset "Linear Gaussian model" begin
        ps, nw = Jaynes.importance_sampling(observations, n_traces, LinearGaussian, (0.0, 1.0))
        @test length(ps.calls) == n_traces
        @test length(ps.lws) == n_traces
        @test isapprox(sum(nw), 1., atol = 1e-9)
        @test !isnan(ps.lmle)
        for call in ps.calls
            @test (call[:z]) == z
        end
    end

    @testset "Linear Gaussian proposal" begin
        ps, nw = Jaynes.importance_sampling(observations, n_traces, LinearGaussian, (0.0, 1.0), LinearGaussianProposal, ())
        @test length(ps.calls) == n_traces
        @test length(ps.lws) == n_traces
        @test isapprox(sum(nw), 1., atol = 1e-9)
        @test !isnan(ps.lmle)
        for call in ps.calls
            @test (call[:z]) == z
        end
    end
end
