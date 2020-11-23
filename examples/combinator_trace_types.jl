module CombinatorTraceTypes

include("../src/Jaynes.jl")
using .Jaynes
using Gen

model1 = @jaynes (i::Int, x::Float64) -> begin
    y ~ Normal(x, 1.0)
    z ~ Normal(y, 3.0)
    z
end (NoPipeline)
display(model1)
uc = Gen.Unfold(model1)

model2 = @jaynes (i::Int, x::Float64) -> begin
    y ~ uc(i, x)
end (DefaultPipeline)
display(model2)

tr = simulate(model2, (10, 5.0))

end # module
