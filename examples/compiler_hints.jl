module CompilerHints

include("../src/Jaynes.jl")
using .Jaynes
using Gen

naive_model = @jaynes function baz(x::Int)
    y = 0.0
    for i in 1 : x
        y += {:y => i} ~ Normal(0.0, 1.0)
    end
end (DefaultPipeline)

ker_model = @jaynes function ker(i::Int, y::Float64)
    y += {:y} ~ Normal(0.0, 1.0)
    y
end
comb = Unfold(ker_model)

model3 = @jaynes function baz(x::Int)
    y = 0.0
    y ~ comb(x, y)
end (DefaultPipeline)

tr = simulate(model3, (5, ))

end # module
