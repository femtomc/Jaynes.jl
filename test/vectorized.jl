function kernel(x::Float64)
    y = rand(:y, Normal(x, 1.0))
    return y
end
test_foldr = () -> foldr(rand, :k, kernel, 10, 1.0)
test_map = () -> map(rand, :k, kernel, [1.0, 2.0, 3.0, 4.0, 5.0])

@testset "Trace" begin
    cl = trace(test_map)
    for i in 1:5
        @test haskey(cl, :k => i => :y)
    end
    cl = trace(test_foldr)
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

#@testset "Update" begin
#    @testset "Vectorized map" begin
#        cl, w = generate(test_map)
#        stored_at_y = cl[:k => 3 => :y]
#        sel = selection((:k => 3 => :y, 5.0))
#        cl, retdiff, d = update(sel, cl)
#        # Discard should be original :x
#        @test d[:k => 3 => :y] == stored_at_y
#        # New should be equal to constraint.
#        @test cl[:k => 3 => :y] == 5.0
#    end
#    
#    @testset "Vectorized foldr" begin
#        cl, w = generate(test_foldr)
#        stored_at_y = cl[:k => 3 => :y]
#        sel = selection((:k => 3 => :y, 5.0))
#        cl, retdiff, d = update(sel, cl)
#        # Discard should be original :x
#        @test d[:k => 3 => :y] == stored_at_y
#        # New should be equal to constraint.
#        @test cl[:k => 3 => :y] == 5.0
#    end
#end

#@testset "Regenerate" begin
#    cl, w = generate(LinearGaussian, 0.5, 3.0)
#    stored_at_x = cl[:x]
#    stored_at_y = cl[:y]
#    sel = selection(:x)
#    cl, retdiff, d = regenerate(sel, cl)
#    # Discard should be original :x
#    @test d.query[:x] == stored_at_x
#    # New should not be equal to original.
#    @test cl[:x] != stored_at_x
#    sel = selection(:y)
#    cl, retdiff, d = regenerate(sel, cl)
#    @test d.query[:y] == stored_at_y
#end
#
#@testset "Propose" begin
#    cl, w = propose(LinearGaussian, 0.5, 3.0)
#    # Propose should track score.
#    @test score(cl) != 0.0
#end
#
#@testset "Score" begin
#    sel = selection([(:x, 5.0), (:y, 10.0)])
#    sc = score(sel, LinearGaussian, 0.5, 3.0)
#    @test sc != 0.0
#end

