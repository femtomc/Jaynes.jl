module Effects

include("../src/Walkman.jl")
using .Walkman
using Distributions
using FunctionalCollections: PersistentVector

function bar(x)
    z = rand(:z, Normal, (x, 1.0))
    return rand(:bar, Normal, (z + x, 1.0))
end

function foo1(v::PersistentVector{Float64})
    x = rand(:foo1, Chorus, (bar, v))
    return x
end

function foo2(init::PersistentVector{Float64})
    x = rand(:x, Wavefolder, (foo1, 5, init))
    return x
end

# wavefolder
ctx, tr = trace(foo2, (PersistentVector([0.5, 3.0]), ))
display(tr)
println(get_retval(tr))

end # module
