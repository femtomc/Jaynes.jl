module AnalysisTest

include("../src/Jaynes.jl")
using .Jaynes

function foo()
    x = rand(:x, Normal, (0.0, 1.0))
    y = rand(:y, Normal, (x, 1.0))
    z = 10
    return x, z
end

code = toplevel_analyze(foo, ())
println(code)

end # module
