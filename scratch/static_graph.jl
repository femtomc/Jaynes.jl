module StaticGraph

include("../src/Jaynes.jl")
using .Jaynes

using MetaGraphs, LightGraphs
using IRTools
using InteractiveUtils: @code_lowered
using GraphPlot
using Compose
import Cairo

function simple(x::Float64)
    y = rand(Normal(x, 5.0))
    z = rand(Normal(y, 1.0))
    q = rand(Normal(z, 5.0))
    return q
end

ir = @code_ir simple(5.0)
lowered = @code_lowered simple(5.0)
println(ir)
println(lowered)
graph = dependency_graph(ir)

println(graph)
labels = [props(graph, i)[:name] for i in vertices(graph)]
draw(PDF("graphs/dependency_graph_irtools.pdf", 16cm, 16cm), gplot(graph, nodelabel = labels, arrowlengthfrac = 0.1, layout=stressmajorize_layout))

end # module
