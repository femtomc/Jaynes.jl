module Regenerate

include("../src/Walkman.jl")
using .Walkman
using Distributions

function bar(x)
    z = rand(:z, Normal, (x, 1.0))
    return rand(:bar, Normal, (z + x, 1.0))
end

ctx, tr = trace(bar, (5.0, ))
display(tr)
tr, score = regenerate(tr, (6.0,), Address[:bar])
display(tr)

end # module
