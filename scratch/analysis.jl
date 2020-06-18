module GraphIRScratch

include("../src/Jaynes.jl")
using Jaynes
using Distributions

function bar(z::Float64)
    z = rand(:z, Normal, (z, 5.0))
    return z
end

function foo(x::Int)
    z = rand(:z, Normal, (0.0, 1.0))
    y = z
    q = rand(:q, Normal, (y, 3.0))
    l = rand(:l, Normal, (q, z))
    r = rand(:r, Normal, (l, y))
    m = rand(:bar, bar, (r, ))
    return rand(:bar, bar, (q, ))
end

# ---- Analysis ----

g = Jaynes.construct_graph(foo, Int)
println(g)

end # module
