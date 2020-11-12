module HierarchicalTraceTypes

include("../src/Jaynes.jl")
using .Jaynes

model1 = @jaynes (x::Float64) -> begin
    y ~ Normal(0.0, 1.0)
end (DefaultPipeline)

model2 = @jaynes (x::Float64) -> begin
    y ~ model1(x)
end (DefaultPipeline)

model3 = @jaynes (x::Float64) -> begin
    y ~ model2(x)
    z ~ model1(x)
    q ~ model2(z)
    q
end (DefaultPipeline)

model4 = @jaynes (x::Float64) -> begin
    y ~ model3(x)
    z ~ model2(x)
    q ~ model3(z)
    q
end (DefaultPipeline)

println(model3 << model4)
println(model4 << model4)

end # module
