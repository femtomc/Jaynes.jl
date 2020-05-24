module Effects

include("../src/Walkman.jl")
using .Walkman
using Distributions

function bar(x)
    z = rand(:z, Normal, (x, 1.0))
    return rand(:bar, Normal, (z + x, 1.0))
end

function foo1(v::Vector{Float64})
    x = rand(:x, Walkman.chorus, (bar, v))
    return x
end

function foo2(v::Float64)
    x = rand(:x, Walkman.wavefolder, (bar, 5, v))
    return x
end

# chorus
ctx, tr = trace(foo1, ([0.5, 5.0, 3.0, 5.0, 7.0],))
display(tr)

# wavefolder
ctx, tr = trace(foo2, (0.0, ))
display(tr)

end # module
