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

@testset "Trace" begin
    tr = Trace()
    tr(LinearGaussian, 0.5, 3.0)
    @test haskey(tr, :x)
    @test haskey(tr, :y)
end

@testset "Generate" begin
    sel = selection((:x, 5.0))
    cl, w = generate(sel, LinearGaussian, 0.5, 3.0)
    @test cl[:x] == 5.0
    sel = selection([(:x, 5.0), (:y, 10.0)])
    cl, w = generate(sel, LinearGaussian, 0.5, 3.0)
    @test cl[:x] == 5.0
    @test cl[:y] == 10.0
end

@testset "Update" begin
    cl, w = generate(LinearGaussian, 0.5, 3.0)
    stored_at_x = cl[:x]
    stored_at_y = cl[:y]
    sel = selection((:x, 5.0))
    cl, retdiff, d = update(sel, cl)
    # Discard should be original :x
    @test d[:x] == stored_at_x
    # New should be equal to constraint.
    @test cl[:x] == 5.0
    sel = selection((:y, 10.0))
    cl, retdiff, d = update(sel, cl)
    @test d[:y] == stored_at_y
    @test cl[:x] == 5.0
    @test cl[:y] == 10.0
end

@testset "Regenerate" begin
    cl, w = generate(LinearGaussian, 0.5, 3.0)
    stored_at_x = cl[:x]
    stored_at_y = cl[:y]
    sel = selection(:x)
    cl, retdiff, d = regenerate(sel, cl)
    # Discard should be original :x
    @test d[:x] == stored_at_x
    # New should not be equal to original.
    @test cl[:x] != stored_at_x
    sel = selection(:y)
    cl, retdiff, d = regenerate(sel, cl)
    @test d[:y] == stored_at_y
end

@testset "Propose" begin
    cl, w = propose(LinearGaussian, 0.5, 3.0)
    # Propose should track score.
    @test get_score(cl) != 0.0
end

@testset "Score" begin
    sel = selection([(:x, 5.0), (:y, 10.0)])
    sc = score(sel, LinearGaussian, 0.5, 3.0)
    @test sc != 0.0
end
