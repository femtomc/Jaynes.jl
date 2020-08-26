function kernel(x::Float64)
    y = rand(:y, Normal(x, 1.0))
    return y
end
test_markov = () -> markov(:k, kernel, 5, 1.0)
test_plate = () -> plate(:k, kernel, [1.0, 2.0, 3.0, 4.0, 5.0])

@testset "Simulate" begin
    @testset "Vectorized plate" begin
        ret, cl = simulate(test_plate)
        for i in 1:5
            @test haskey(cl, (:k, i, :y))
        end
    end
    @testset "Vectorized markov" begin
        ret, cl = simulate(test_markov)
        for i in 1:5
            @test haskey(cl, (:k, i, :y))
        end
    end
end

@testset "Constrained generate" begin
    sel = target([(:k, 3, :y) => 5.0])
    @testset "Vectorized plate" begin
        ret, cl, _ = generate(sel, test_plate)
        for i in 1:5
            @test haskey(cl, (:k, i, :y))
        end
        @test (cl[:k, 3, :y]) == 5.0
    end
    @testset "Vectorized markov" begin
        ret, cl, _ = generate(sel, test_markov)
        for i in 1:5
            @test haskey(cl, (:k, i, :y))
        end
        @test (cl[:k, 3, :y]) == 5.0
    end
end

@testset "Update" begin
    @testset "Vectorized plate" begin
        ret, cl = simulate(test_plate)
        original_score = get_score(cl)
        stored_at_y = (cl[:k, 3, :y])
        sel = target([(:k, 3, :y) => 5.0])
        ret, cl, w, rd, d = update(sel, cl)
        @test get_score(cl) - w ≈ original_score
    end

    @testset "Vectorized markov" begin
        ret, cl = simulate(test_markov)
        stored_at_y = (cl[:k, 3, :y])
        sel = target([(:k, 3, :y) => 5.0])
        ret, cl, w, retdiff, d = update(sel, cl)
        @test (cl[:k, 3, :y]) == 5.0
    end
end

@testset "Regenerate" begin
    @testset "Vectorized plate" begin
        ret, cl = simulate(test_plate)
        original_score = get_score(cl)
        stored = cl[:k, 3, :y]
        sel = target([(:k, 3, :y)])
        ret, cl, w, retdiff, d = regenerate(sel, cl)
        @test cl[:k, 3, :y] != stored
        @test get_score(cl) - w ≈ original_score
    end

    @testset "Vectorized markov" begin
        ret, cl = simulate(test_markov)
        original_score = get_score(cl)
        stored = (cl[:k, 3, :y])
        sel = target([(:k, 3, :y)])
        ret, cl, w, retdiff, d = regenerate(sel, cl)
        @test (cl[:k, 3, :y]) != stored
        @test get_score(cl) - w ≈ original_score
    end
end

@testset "Propose" begin
    @testset "Vectorized plate" begin
        ret, cl, w = propose(test_plate)
        @test w != 0.0
    end
    @testset "Vectorized markov" begin
        ret, cl, w = propose(test_markov)
        @test w != 0.0
    end
end

@testset "Score" begin
    sel = target([(:k, 1, :y) => 5.0,
                  (:k, 2, :y) => 5.0, 
                  (:k, 3, :y) => 5.0,
                  (:k, 4, :y) => 5.0,
                  (:k, 5, :y) => 5.0])
    @testset "Vectorized plate" begin
        ret, sc = score(sel, test_plate)
        @test sc != 0.0
    end
    @testset "Vectorized markov" begin
        ret, sc = score(sel, test_markov)
        @test sc != 0.0
    end
end
