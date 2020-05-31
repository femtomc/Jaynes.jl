module Profiling

include("../src/Jaynes.jl")
using .Jaynes
using Distributions

function foo1()
    x = rand(:x, Normal, (3.0, 1.0))
    y = rand(:y, Normal, (x + 15.0, 1.0))
    return y
end

ctx = inference_compilation(foo1, (), Dict{Address, Float64}())

end # module
