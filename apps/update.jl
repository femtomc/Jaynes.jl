module Regenerate

include("../src/Walkman.jl")
using .Walkman
using Distributions

function bar(x)
    z = rand(:z, Normal, (x, 1.0))
    return rand(:bar, Normal, (z + x, 1.0))
end

tr, score = trace(bar, (5.0, ))
display(tr)
obs = map(collect(tr.chm)) do (k, v)
    (k, v.val)
end
tr, score = update(tr, (5.0,), constraints([obs[1]]))
display(tr)

end # module
