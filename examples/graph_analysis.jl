module GraphAnalysis

include("../src/Jaynes.jl")
using .Jaynes

model = () -> begin
    x = rand(:x, Normal(0.0, 1.0))
    y = rand(:y, Normal(x, 5.0))
    z = rand(:z, Normal(x, 1.0))
    q = rand(:q, Normal(y + z, 5.0))
    q
end

g = construct_graph(model)
display(g)

end # module
