module RecursiveTraceTypes

include("../src/Jaynes.jl")
using .Jaynes

model1 = @jaynes (x::Float64) -> begin
    y ~ Normal(0.0, 1.0)
end (check)

model2 = @jaynes (x::Float64) -> begin
    y ~ model1(x)
end (check)

model3 = @jaynes (x::Float64) -> begin
    y ~ model2(x)
    z ~ model1(x)
    q ~ model2(x)
    q
end (check)
display(model3)

end # module
