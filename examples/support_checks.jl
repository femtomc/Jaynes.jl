module SupportChecking

include("../src/Jaynes.jl")
using .Jaynes

#model1 = @jaynes (check) x::Int -> begin
#    x > 10 ? rand(:y, Normal(0.0, 1.0)) : rand(:y, Bernoulli(0.5))
#end

model2 = @jaynes (check) x::Int -> begin
    rand(:y, Normal(0.0, 1.0))
    rand(:y, Bernoulli(0.5))
end

end # module
