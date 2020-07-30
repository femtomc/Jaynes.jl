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

@testset "Constrained selections" begin

    @testset "Constrained by address" begin
        sel = selection([(:x, ) => 5.0,
                         (:x, :q, :z) => 10.0])
        @test sel == sel
        @test haskey(sel, (:x, ))
        @test haskey(sel, :x)
        @test haskey(sel, (:x, :q, :z))
        @test sel[:x, :q, :z] == 10.0

        sel2 = selection([(:z, :m) => 15.0])
        merge!(sel, sel2)
        @test sel == selection([(:x, ) => 5.0,
                                (:x, :q, :z) => 10.0,
                                (:z, :m) => 15.0])
        arr = array(sel, Float64)
        @test arr == [5.0, 10.0, 15.0]
        back = selection(sel, arr)
        @test back == sel
    end
    
    @testset "Unconstrained by address" begin
        sel = selection([(:x, ), (:x, :q, :z)])
        @test sel == sel
        @test haskey(sel, (:x, ))
        @test haskey(sel, :x)
        @test haskey(sel, (:x, :q, :z))

        sel2 = selection([(:z, :m)])
        merge!(sel, sel2)
        @test sel == selection([(:x, ), (:x, :q, :z), (:z, :m)])
    end

    @testset "Anywhere" begin
        anyw = anywhere([(:x, ) => 5.0, 
                         (:q => 21, ) => 10.0])
        ret, cl, w = generate(anyw, AnywhereTopLevel)
        @test get_ret(cl[:x]) == 5.0
        @test get_ret(cl[:y, :x]) == 5.0
        @test get_ret(cl[:y, :y, :x]) == 5.0
        @test get_ret(cl[:y, :y, :loop, :q => 21]) == 10.0
    end

    @testset "Filtering" begin
        observations = selection([(:x, ) => 5.0, 
                                  (:z, :x) => 5.0, 
                                  (:z, :z, :y) => 5.0])
        filtered = filter(x -> x == :y, x -> true, observations)
        @test has_query(filtered, (:z, :z, :y))
        filtered = filter(x -> x == :x, x -> true, observations)
        @test !has_query(filtered, (:z, :z, :y))
        @test has_query(filtered, :x)
        @test has_query(filtered, (:z, :x))
    end
end
