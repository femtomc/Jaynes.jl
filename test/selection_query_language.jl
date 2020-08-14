function AnywhereLevelTwo()
    x = rand(:x, Normal(0.0, 1.0))
    rand(:loop, AnywhereInLoop)
    return x
end

function AnywhereLevelOne()
    x = rand(:x, Normal(0.0, 1.0))
    y = rand(:y, AnywhereLevelTwo)
    return x
end

function AnywhereTopLevel()
    x = rand(:x, Normal(0.0, 1.0))
    y = rand(:y, AnywhereLevelOne)
    return x
end

function AnywhereInLoop()
    x = rand(:q, Normal(0.0, 1.0))
    for i in 1:50
        z = rand(:q => i, Normal(0.0, 1.0))
    end
end

@testset "Constrained targets" begin

    @testset "Constrained by address" begin
        sel = target([(:x, ) => 5.0,
                      (:y, :q, :z) => 10.0])
        @test sel == sel
        @test haskey(sel, (:x, ))
        @test haskey(sel, :x)
        @test haskey(sel, (:y, :q, :z))
        @test sel[:y, :q, :z] == 10.0

        sel2 = target([(:z, :m) => 15.0])
        merge!(sel, sel2)
        @test sel == target([(:x, ) => 5.0,
                             (:m, :q, :z) => 10.0,
                             (:z, :m) => 15.0])
        arr = array(sel, Float64)
        @test arr == [5.0, 10.0, 15.0]
        back = target(sel, arr)
        @test back == sel
    end

    @testset "Unconstrained by address" begin
        sel = target([(:x, ), (:x, :q, :z)])
        @test sel == sel
        @test haskey(sel, (:x, ))
        @test haskey(sel, :x)
        @test haskey(sel, (:x, :q, :z))

        sel2 = target([(:z, :m)])
        merge!(sel, sel2)
        @test sel == target([(:x, ), (:x, :q, :z), (:z, :m)])
    end

    #@testset "Filtering" begin
    #    observations = target([(:x, ) => 5.0, 
    #                           (:z, :x) => 5.0, 
    #                           (:z, :z, :y) => 5.0])
    #    filtered = filter(x -> x == :y, x -> true, observations)
    #    @test has_top(filtered, (:z, :z, :y))
    #    filtered = filter(x -> x == :x, x -> true, observations)
    #    @test !has_top(filtered, (:z, :z, :y))
    #    @test has_top(filtered, :x)
    #    @test has_top(filtered, (:z, :x))
    #end
end
