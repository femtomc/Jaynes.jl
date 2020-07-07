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

    @testset "Anywhere" begin
        observations = Jaynes.ConstrainedHierarchicalSelection()
        anywhere = Jaynes.ConstrainedAnywhereSelection([(:x, 5.0), (:q => 21, 10.0)])
        un = Jaynes.union(observations, anywhere)
        ctx = Generate(Trace(), un)
        call = ctx(AnywhereTopLevel)
        @test ctx.tr[:x] == 5.0
        @test ctx.tr[:y => :x] == 5.0
        @test ctx.tr[:y => :y => :x] == 5.0
        @test ctx.tr[:y => :y => :loop => :q => 21] == 10.0
    end
    
    @testset "Filtering" begin
        observations = selection([(:x, 5.0), (:z => :x, 5.0), (:z => :z => :y, 5.0)])
        filtered = filter([:y], observations)
        display(filtered)
    end
end
