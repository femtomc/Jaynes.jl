jmodel = @jaynes () -> begin
    x ~ Normal(0.0, 1.0)
    y ~ Normal(x, 0.05)
end

jproposal = @jaynes obs -> begin
    x ~ Normal(obs, 1.0)
end
    
obs = choicemap([(:y, ) => 1.0])

@testset "Simulate" begin
    tr = simulate(jmodel, ())
    @test has_value(tr, :x)
    @test has_value(tr, :y)
end

@testset "Generate" begin
    @test has_value(obs, :y)
    tr, w = generate(jmodel, (), obs)
    @test has_value(tr, :x)
    @test has_value(tr, :y)
    @test get_value(tr, :y) == 1.0
end

@testset "Importance sampling" begin
    trs, lnw, lmle = Gen.importance_sampling(jmodel, (), obs, jproposal, (obs[:y], ), 5000)
    est = sum(map(zip(trs, lnw)) do (tr, w)
            tr[:x] * exp(w)
        end)
    @test est ≈ 1.0 atol = 1e-2
end
