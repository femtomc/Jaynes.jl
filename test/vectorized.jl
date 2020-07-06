function kernel(x::Float64)
    y = rand(:y, Normal(x, 1.0))
    return y
end
test_foldr = () -> foldr(rand, :k, kernel, 10, 1.0)
test_map = () -> map(rand, :k, kernel, [1.0, 2.0, 3.0, 4.0, 5.0])

@testset "Unconstrained generate" begin
    cl, _ = generate(test_map)
    for i in 1:5
        @test haskey(cl, :k => i => :y)
    end
    cl, _ = generate(test_foldr)
    for i in 1:10
        @test haskey(cl, :k => i => :y)
    end
end

@testset "Constrained generate" begin
    sel = selection((:k => 3 => :y, 5.0))
    cl, _ = generate(sel, test_map)
    for i in 1:5
        @test haskey(cl, :k => i => :y)
    end
    @test cl[:k => 3 => :y] == 5.0
    cl, _ = generate(sel, test_foldr)
    for i in 1:10
        @test haskey(cl, :k => i => :y)
    end
    @test cl[:k => 3 => :y] == 5.0
end

@testset "Update" begin
end

@testset "Regenerate" begin
end

@testset "Propose" begin
end

@testset "Score" begin
end
