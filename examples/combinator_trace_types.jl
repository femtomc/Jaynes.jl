module CombinatorTraceTypes

include("../src/Jaynes.jl")
using .Jaynes
using Gen

model1 = @jaynes (i::Int, x::Float64) -> begin
    y ~ Normal(x, 1.0)
    z ~ Normal(y, 3.0)
    z
end (check)
unfold = Gen.Unfold(model1)

model2 = @jaynes (i::Int, x::Float64) -> begin
    y ~ unfold(i, x)
end (check)
display(model2)

end # module
