function kernel(x::Float64)
    y = rand(:y, Normal(x, 1.0))
    return y
end
test_markov = () -> markov(:k, kernel, 5, 1.0)
test_plate = () -> plate(:k, kernel, [1.0, 2.0, 3.0, 4.0, 5.0])

@testset "Simulate" begin
    ret, cl = simulate(test_plate)
    display(cl)
    for i in 1:5
        @test haskey(cl, :k => i => :y)
    end
    ret, cl = simulate(test_markov)
    for i in 1:5
        @test haskey(cl, :k => i => :y)
    end
end

@testset "Constrained generate" begin
    sel = selection((:k => 3 => :y, 5.0))
    ret, cl, _ = generate(sel, test_plate)
    for i in 1:5
        @test haskey(cl, :k => i => :y)
    end
    @test cl[:k => 3 => :y] == 5.0
    ret, cl, _ = generate(sel, test_markov)
    for i in 1:5
        @test haskey(cl, :k => i => :y)
    end
    @test cl[:k => 3 => :y] == 5.0
end

#@testset "Update" begin
#    @testset "Vectorized plate" begin
#        cl, w = generate(test_plate)
#        stored_at_y = cl[:k => 3 => :y]
#        sel = selection((:k => 3 => :y, 5.0))
#        cl, retdiff, d = update(sel, cl)
#        # Discard should be original :x
#        @test d[:k => 3 => :y] == stored_at_y
#        # New should be equal to constraint.
#        @test cl[:k => 3 => :y] == 5.0
#    end
#    
#    @testset "Vectorized markov" begin
#        cl, w = generate(test_markov)
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

@testset "Propose" begin
    ret, cl, w = propose(test_markov)
    @test get_score(cl) != 0.0
    ret, cl, w = propose(test_plate)
    @test get_score(cl) != 0.0
end

@testset "Score" begin
    sel = selection([(:k => 1 => :y, 5.0), 
                     (:k => 2 => :y, 5.0), 
                     (:k => 3 => :y, 5.0),
                     (:k => 4 => :y, 5.0),
                     (:k => 5 => :y, 5.0)])
    ret, sc = score(sel, test_markov)
    @test sc != 0.0
    ret, sc = score(sel, test_plate)
    @test sc != 0.0
end

