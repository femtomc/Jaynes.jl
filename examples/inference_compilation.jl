module Profiling

include("../src/Jaynes.jl")
using .Jaynes
using Distributions

function foo1()
    x = rand(:x, Normal, (3.0, 10.0))
    y = rand(:y, Normal, (x + 15.0, 6.0))
    return y
end

ctx = inference_compilation(foo1, (), :y)

end # module
